import tempfile
from math import log

from cobrak.constants import ALL_OK_KEY, OBJECTIVE_VAR_NAME
from cobrak.dataclasses import ExtraLinearConstraint
from cobrak.example_models import toy_model
from cobrak.io import (
    load_annotated_sbml_model_as_cobrak_model,
    save_cobrak_model_as_annotated_sbml_model,
)
from cobrak.lps import perform_lp_variability_analysis
from cobrak.nlps import perform_nlp_reversible_optimization
from cobrak.standard_solvers import SCIP

with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as temp_sbml_file:
    save_cobrak_model_as_annotated_sbml_model(toy_model, filepath=temp_sbml_file.name)
    model = load_annotated_sbml_model_as_cobrak_model(filepath=temp_sbml_file.name)

model.extra_linear_constraints = [
    ExtraLinearConstraint(
        stoichiometries={"x_ATP": 1.0, "x_ADP": -1.0},
        lower_value=log(3.0),
    )
]

variability_dict = perform_lp_variability_analysis(
    model,
    with_enzyme_constraints=True,
    with_thermodynamic_constraints=True,
    min_flux_cutoff=1e-7,
)

res_minlp = perform_nlp_reversible_optimization(
    cobrak_model=model,
    objective_target="ATP_Consumption",
    objective_sense=+1,
    variability_dict=variability_dict,
    with_kappa=True,
    with_gamma=True,
    with_alpha=False,
    with_iota=False,
    solver=SCIP,
)
print("MINLP ATP_Consumption =", res_minlp.get(OBJECTIVE_VAR_NAME))
print("MINLP ALL_OK =", res_minlp.get(ALL_OK_KEY))

print("DONE")
