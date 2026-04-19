from pyomo.core import ConcreteModel, exp
from pyomo.core import log as ln
from pyomo.environ import Constraint, Objective, Reals, Var, maximize  # noqa: F401

from cobrak.example_models import toy_model
from cobrak.pyomo_functionality import get_solver
from cobrak.standard_solvers import SCIP

QUASI_INF = 1e6


def create_stoichiometric_matrix(_toy_model) -> dict[str, dict[str, float]]:
    """Build a metabolite->reaction stoichiometry lookup from COBRA-k reaction data."""
    matrix: dict[str, dict[str, float]] = {
        met_id: {rxn_id: 0.0 for rxn_id in _toy_model.reactions}
        for met_id in _toy_model.metabolites
    }
    for rxn_id, reaction in _toy_model.reactions.items():
        for met_id, coeff in reaction.stoichiometries.items():
            if met_id in matrix:
                matrix[met_id][rxn_id] = coeff
    return matrix


def _reaction_has_kinetics(_toy_model, rxn_id: str) -> bool:
    enzyme_data = _toy_model.reactions[rxn_id].enzyme_reaction_data
    return (enzyme_data is not None) and (enzyme_data.k_cat < 1e19)


def _reaction_has_thermo(_toy_model, rxn_id: str) -> bool:
    return _toy_model.reactions[rxn_id].dG0 is not None


def _get_reaction_enzyme_mw(_toy_model, rxn_id: str) -> float:
    reaction = _toy_model.reactions[rxn_id]
    enzyme_data = reaction.enzyme_reaction_data
    if enzyme_data is None:
        return 0.0
    return sum(
        _toy_model.enzymes[enz_id].molecular_weight for enz_id in enzyme_data.identifiers
    )


def _get_reaction_enzyme_bounds(_toy_model, rxn_id: str) -> tuple[float, float]:
    reaction = _toy_model.reactions[rxn_id]
    enzyme_data = reaction.enzyme_reaction_data
    if enzyme_data is None:
        return (0.0, 0.0)

    min_conc = 0.0
    max_conc = QUASI_INF
    has_specific_max = False
    for enz_id in enzyme_data.identifiers:
        enzyme = _toy_model.enzymes[enz_id]
        if enzyme.min_conc is not None:
            min_conc = max(min_conc, enzyme.min_conc)
        if enzyme.max_conc is not None:
            max_conc = min(max_conc, enzyme.max_conc)
            has_specific_max = True

    # If no reaction-specific max is given, use protein-pool implied upper bound.
    if not has_specific_max:
        full_mw = _get_reaction_enzyme_mw(_toy_model, rxn_id)
        if full_mw > 0.0:
            max_conc = min(max_conc, _toy_model.max_prot_pool / full_mw)

    return (min_conc, max_conc)


def create_base_model(_toy_model) -> ConcreteModel:
    reactions = _toy_model.reactions
    metabolites = _toy_model.metabolites
    stoichiometric_matrix = create_stoichiometric_matrix(_toy_model)

    model = ConcreteModel()
    model.stoichiometric_matrix = stoichiometric_matrix

    def reaction_flux_bounds(_m, rxn_id):
        reaction = reactions[rxn_id]
        return (reaction.min_flux, reaction.max_flux)

    model.reaction_fluxes = Var(
        reactions.keys(),
        domain=Reals,
        bounds=reaction_flux_bounds,
    )

    def concentration_bounds(_m, met_id):
        metabolite = metabolites[met_id]
        return (metabolite.log_min_conc, metabolite.log_max_conc)

    model.metabolite_log_concentrations = Var(
        metabolites.keys(),
        domain=Reals,
        bounds=concentration_bounds,
    )

    def enzyme_bounds(_m, rxn_id):
        return _get_reaction_enzyme_bounds(_toy_model, rxn_id)

    model.enzyme_reaction_concentrations = Var(
        reactions.keys(),
        domain=Reals,
        bounds=enzyme_bounds,
    )

    model.reactions_substrate_tilde = Var(
        reactions.keys(),
        domain=Reals,
        bounds=(-QUASI_INF, QUASI_INF),
    )
    model.reactions_product_tilde = Var(
        reactions.keys(),
        domain=Reals,
        bounds=(-QUASI_INF, QUASI_INF),
    )
    model.reactions_kappa = Var(
        reactions.keys(),
        domain=Reals,
        bounds=(0.0, 1.0),
    )
    model.reactions_driving_force = Var(
        reactions.keys(),
        domain=Reals,
        bounds=(-QUASI_INF, QUASI_INF),
    )
    model.reactions_gamma = Var(
        reactions.keys(),
        domain=Reals,
        bounds=(0.0, 1.0),
    )

    # Scalar MDF variable that can be optimized.
    model.B = Var(domain=Reals, bounds=(-QUASI_INF, QUASI_INF))

    return model


def build_steady_state_model(opt_model: ConcreteModel, _toy_model) -> ConcreteModel:
    stoichiometric_matrix = opt_model.stoichiometric_matrix

    def steady_state_rule(model, met_id):
        return (
            sum(
                model.reaction_fluxes[rxn_id] * stoichiometric_matrix[met_id][rxn_id]
                for rxn_id in _toy_model.reactions
            )
            == 0.0
        )

    opt_model.steady_state_constraints = Constraint(
        _toy_model.metabolites.keys(),
        rule=steady_state_rule,
    )
    return opt_model


def set_enzyme_constraints(
    opt_model: ConcreteModel,
    _toy_model,
    total_enzyme_concentration: float,
) -> ConcreteModel:
    def enzyme_concentration_rule(model):
        return (
            sum(
                model.enzyme_reaction_concentrations[rxn_id]
                * _get_reaction_enzyme_mw(_toy_model, rxn_id)
                for rxn_id in _toy_model.reactions
                if _reaction_has_kinetics(_toy_model, rxn_id)
            )
            <= total_enzyme_concentration
        )

    opt_model.enzyme_concentration_constraint = Constraint(rule=enzyme_concentration_rule)
    return opt_model


def set_kappa_constraints(
    opt_model: ConcreteModel,
    _toy_model,
    EPSILON_KAPPA=1e-4,
) -> ConcreteModel:
    stoichiometric_matrix = opt_model.stoichiometric_matrix

    def substrate_rule(model, rxn_id):
        if not _reaction_has_kinetics(_toy_model, rxn_id):
            return model.reactions_substrate_tilde[rxn_id] == 0.0

        rxn_mets = [
            met_id
            for met_id in _toy_model.metabolites
            if stoichiometric_matrix[met_id][rxn_id] < 0
        ]
        reaction = _toy_model.reactions[rxn_id]
        km_dict = reaction.enzyme_reaction_data.k_ms

        return model.reactions_substrate_tilde[rxn_id] == sum(
            abs(stoichiometric_matrix[met_id][rxn_id])
            * model.metabolite_log_concentrations[met_id]
            for met_id in rxn_mets
        ) - sum(
            abs(stoichiometric_matrix[met_id][rxn_id]) * ln(km_dict.get(met_id, 1.0))
            for met_id in rxn_mets
        )

    def product_rule(model, rxn_id):
        if not _reaction_has_kinetics(_toy_model, rxn_id):
            return model.reactions_product_tilde[rxn_id] == 0.0

        rxn_mets = [
            met_id
            for met_id in _toy_model.metabolites
            if stoichiometric_matrix[met_id][rxn_id] > 0
        ]
        reaction = _toy_model.reactions[rxn_id]
        km_dict = reaction.enzyme_reaction_data.k_ms

        return model.reactions_product_tilde[rxn_id] >= sum(
            abs(stoichiometric_matrix[met_id][rxn_id])
            * model.metabolite_log_concentrations[met_id]
            for met_id in rxn_mets
        ) - sum(
            abs(stoichiometric_matrix[met_id][rxn_id]) * ln(km_dict.get(met_id, 1.0))
            for met_id in rxn_mets
        )

    def kappa_rule(model, rxn_id):
        if not _reaction_has_kinetics(_toy_model, rxn_id):
            return model.reactions_kappa[rxn_id] == 1.0

        return model.reactions_kappa[rxn_id] <= (
            EPSILON_KAPPA
            + (
                exp(model.reactions_substrate_tilde[rxn_id])
                / (
                    1
                    + exp(model.reactions_substrate_tilde[rxn_id])
                    + exp(model.reactions_product_tilde[rxn_id])
                )
            )
        )

    opt_model.substrate_constraints = Constraint(
        _toy_model.reactions.keys(), rule=substrate_rule
    )
    opt_model.product_constraints = Constraint(_toy_model.reactions.keys(), rule=product_rule)
    opt_model.kappa_constraints = Constraint(_toy_model.reactions.keys(), rule=kappa_rule)
    return opt_model


def set_gamma_constraints(
    opt_model: ConcreteModel,
    _toy_model,
    EPSILON_GAMMA=1e-4,
) -> ConcreteModel:
    stoichiometric_matrix = opt_model.stoichiometric_matrix

    def driving_force_rule(model, rxn_id):
        reaction = _toy_model.reactions[rxn_id]
        if reaction.dG0 is None:
            return model.reactions_driving_force[rxn_id] == 0.0

        return model.reactions_driving_force[
            rxn_id
        ] <= -reaction.dG0 - _toy_model.R * _toy_model.T * sum(
            model.metabolite_log_concentrations[met_id]
            * stoichiometric_matrix[met_id][rxn_id]
            for met_id in reaction.stoichiometries
        )

    def gamma_rule(model, rxn_id):
        if not _reaction_has_thermo(_toy_model, rxn_id):
            return model.reactions_gamma[rxn_id] == 1.0

        return model.reactions_gamma[rxn_id] <= (
            EPSILON_GAMMA
            + (
                1
                - exp(-model.reactions_driving_force[rxn_id] / (_toy_model.R * _toy_model.T))
            )
        )

    opt_model.driving_force_constraints = Constraint(
        _toy_model.reactions.keys(), rule=driving_force_rule
    )
    opt_model.gamma_constraints = Constraint(_toy_model.reactions.keys(), rule=gamma_rule)
    return opt_model


def set_flux_constraints(opt_model: ConcreteModel, _toy_model) -> ConcreteModel:
    # v_i <= E_i * kcat_i * kappa_i * gamma_i
    def flux_constraint_rule(model, rxn_id):
        if not _reaction_has_kinetics(_toy_model, rxn_id):
            return Constraint.Skip

        k_cat = _toy_model.reactions[rxn_id].enzyme_reaction_data.k_cat
        return model.reaction_fluxes[rxn_id] <= (
            model.enzyme_reaction_concentrations[rxn_id]
            * k_cat
            * model.reactions_kappa[rxn_id]
            * model.reactions_gamma[rxn_id]
        )

    opt_model.flux_constraints = Constraint(
        _toy_model.reactions.keys(), rule=flux_constraint_rule
    )
    return opt_model


def set_max_metabolite_concentration(
    opt_model: ConcreteModel,
    _toy_model,
    TOTAL_METABOLITE_CONCENTRATION=1.0,
) -> ConcreteModel:
    # sum_j w_j * exp(log_conc_j) <= total_concentration
    def molar_concentration_rule(model):
        return (
            sum(
                ((_toy_model.metabolites[met_id].molar_mass or 1.0) / 1000.0)
                * exp(model.metabolite_log_concentrations[met_id])
                for met_id in _toy_model.metabolites
            )
            <= TOTAL_METABOLITE_CONCENTRATION
        )

    opt_model.molar_concentration_constraint = Constraint(rule=molar_concentration_rule)
    return opt_model


def set_min_driving_force_gamma_constraints(
    opt_model: ConcreteModel,
    _toy_model,
    MIN_DRIVING_FORCE,
) -> ConcreteModel:
    thermo_reactions = [
        rxn_id for rxn_id in _toy_model.reactions if _reaction_has_thermo(_toy_model, rxn_id)
    ]

    def min_driving_force_rule(model, rxn_id):
        return model.B <= model.reactions_driving_force[rxn_id]

    def B_rule(model):
        return model.B >= MIN_DRIVING_FORCE

    opt_model.min_driving_force_constraints = Constraint(
        thermo_reactions, rule=min_driving_force_rule
    )
    opt_model.B_constraint = Constraint(rule=B_rule)
    return opt_model


def build_NLP_without_binary_constraints(_toy_model):
    TOTAL_METABOLITE_CONCENTRATION = 1.0
    TOTAL_ENZYME_CONCENTRATION = _toy_model.max_prot_pool
    MIN_DRIVING_FORCE = 1e-3
    EPSILON_KAPPA = 1e-4
    EPSILON_GAMMA = 1e-4

    model = create_base_model(_toy_model)
    model = build_steady_state_model(model, _toy_model)
    model = set_enzyme_constraints(model, _toy_model, TOTAL_ENZYME_CONCENTRATION)
    model = set_kappa_constraints(model, _toy_model, EPSILON_KAPPA=EPSILON_KAPPA)
    model = set_gamma_constraints(model, _toy_model, EPSILON_GAMMA=EPSILON_GAMMA)
    model = set_flux_constraints(model, _toy_model)
    model = set_max_metabolite_concentration(
        model,
        _toy_model,
        TOTAL_METABOLITE_CONCENTRATION,
    )
    model = set_min_driving_force_gamma_constraints(model, _toy_model, MIN_DRIVING_FORCE)
    return model


def parse_minimal_results(
    model: ConcreteModel, _toy_model
) -> dict[str, dict[str, float | None] | float | None]:
    """Extract core values after solve without using COBRA-k solution utilities."""
    result: dict[str, dict[str, float | None] | float | None] = {
        "B": model.B.value,
        "reaction_fluxes": {},
        "enzyme_reaction_concentrations": {},
        "reactions_gamma": {},
        "reactions_driving_force": {},
    }
    for rxn_id in _toy_model.reactions:
        result["reaction_fluxes"][rxn_id] = model.reaction_fluxes[rxn_id].value
        result["enzyme_reaction_concentrations"][rxn_id] = (
            model.enzyme_reaction_concentrations[rxn_id].value
        )
        result["reactions_gamma"][rxn_id] = model.reactions_gamma[rxn_id].value
        result["reactions_driving_force"][rxn_id] = model.reactions_driving_force[
            rxn_id
        ].value
    return result


if __name__ == "__main__":
    m = build_NLP_without_binary_constraints(toy_model)
    m.obj = Objective(expr=m.reaction_fluxes["ATP_Consumption"], sense=maximize)

    solver = get_solver(SCIP)
    solver.solve(m, tee=True)

    result = parse_minimal_results(m, toy_model)
    print("B:", result["B"])
    print("ATP_Consumption:", result["reaction_fluxes"]["ATP_Consumption"])
