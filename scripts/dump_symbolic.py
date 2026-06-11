"""
Dump all symbolic equations generated from a model specification file.

Usage:
    python dump_symbolic.py <model_spec_file> [output_file]

If output_file is omitted, writes to <model_spec_basename>_symbolic.txt
in the same directory as the model spec.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sympy import symbols, diff, Matrix, Poly, simplify
from linkage.symbolic.polynomial import BindingPolynomial
from linkage.symbolic.model import SymbolicBindingModel


def dump_symbolic(spec_path, output_path=None):
    """Read a model spec and write all symbolic equations to a text file."""

    with open(spec_path) as f:
        model_spec = f.read()

    if output_path is None:
        base, _ = os.path.splitext(spec_path)
        output_path = f"{base}_symbolic.txt"

    # Build the full symbolic model (including Jacobian)
    sbm = SymbolicBindingModel(model_spec, debug=False, use_symbolic_jacobian=True)
    poly = sbm.physical_poly

    lines = []

    def section(title):
        lines.append("")
        lines.append("=" * 70)
        lines.append(title)
        lines.append("=" * 70)

    def subsection(title):
        lines.append("")
        lines.append(f"--- {title} ---")

    # ── Header ──
    lines.append("SYMBOLIC MODEL DUMP")
    lines.append(f"Source: {os.path.abspath(spec_path)}")
    lines.append("")
    lines.append("Model specification:")
    for line in model_spec.strip().splitlines():
        lines.append(f"  {line}")

    # ── 1. Parsed model ──
    section("1. PARSED MODEL")

    subsection("Equilibria")
    for k, (reactants, products) in poly._equilibria.items():
        lines.append(f"  {' + '.join(reactants)} -> {' + '.join(products)} ; {k}")

    subsection("Mass conservation (species block)")
    for macro, (micros, stoichs) in poly._species.items():
        terms = []
        for m, s in zip(micros, stoichs):
            terms.append(f"{s}*{m}" if s != 1 else m)
        lines.append(f"  {macro} = {' + '.join(terms)}")

    subsection("Identified species")
    lines.append(f"  Free monomer (polynomial variable): {poly._c_species_name}")
    lines.append(f"  Total concentration:                {poly._ct_macrospecies_name}")
    lines.append(f"  Micro-species: {', '.join(poly._micro_species)}")
    lines.append(f"  Macro-species: {', '.join(poly._macro_species)}")
    lines.append(f"  Equilibrium constants: {', '.join(poly._constants)}")

    # ── 2. Reparameterization ──
    if sbm.reparam_rules:
        section("2. REPARAMETERIZATION RULES")
        lines.append(f"  Regression parameters: {', '.join(sbm.regression_params)}")
        lines.append(f"  Physical parameters:   {', '.join(sbm.all_physical_params)}")
        subsection("Rules (physical = f(regression))")
        for sym, expr in sbm.reparam_rules.items():
            lines.append(f"  {sym} = {expr}")

        subsection("Parameter mapper Jacobian d(Physical)/d(Regression)")
        mapper = sbm.mapper
        for i, phys in enumerate(mapper.physical_params):
            for j, reg in enumerate(mapper.regression_params):
                entry = mapper.jacobian_matrix[i, j]
                if entry != 0:
                    lines.append(f"  d({phys})/d({reg}) = {entry}")
    else:
        section("2. REPARAMETERIZATION")
        lines.append("  (none — all physical parameters are regression parameters)")
        lines.append(f"  Regression parameters: {', '.join(sbm.regression_params)}")
        lines.append(f"  Physical parameters:   {', '.join(sbm.all_physical_params)}")

    # ── 3. Equilibrium equations ──
    section("3. RAW EQUILIBRIUM EQUATIONS")
    lines.append("  Product = K * Reactant1 * Reactant2 * ...")
    for product_sym, rhs in poly.equilibrium_eqs.items():
        lines.append(f"  {product_sym} = {rhs}")

    # ── 4. Simplified equilibria ──
    section("4. SIMPLIFIED SPECIES EXPRESSIONS")
    lines.append("  (after iterative substitution, all in terms of base species + K)")
    for product_sym, rhs in poly.simplified_eqs.items():
        lines.append(f"  {product_sym} = {rhs}")

    # ── 5. Solved base variables ──
    section("5. SOLVED BASE VARIABLES")
    lines.append("  (from non-CT conservation equations, solved for free base species)")
    if poly.solved_vars:
        for base_sym, expr in poly.solved_vars.items():
            lines.append(f"  {base_sym} = {expr}")
    else:
        lines.append("  (none — only one conservation equation)")

    # ── 6. Final rational equation ──
    section("6. FINAL RATIONAL EQUATION")
    lines.append(f"  Setting CT conservation to zero after substituting all solved vars:")
    lines.append(f"  0 = {poly.final_rational_eq}")

    # ── 7. Binding polynomial ──
    section("7. BINDING POLYNOMIAL P(C) = 0")
    lines.append(f"  (numerator after clearing denominators)")
    lines.append(f"  P({poly._c_species_name}) = {poly.binding_polynomial}")

    subsection(f"Coefficients (descending powers of {poly._c_species_name})")
    coeffs = poly.get_polynomial_coefficients()
    degree = len(coeffs) - 1
    lines.append(f"  Polynomial degree: {degree}")
    for i, coeff in enumerate(coeffs):
        power = degree - i
        lines.append(f"  {poly._c_species_name}^{power}: {coeff}")

    # ── 8. Species expressions in terms of C ──
    section("8. SPECIES CONCENTRATIONS AS FUNCTIONS OF C")
    lines.append(f"  (each micro-species expressed in terms of free {poly._c_species_name} and params)")
    for s_name in poly._micro_species:
        expr = sbm.species_exprs.get(s_name, "?")
        lines.append(f"  [{s_name}] = {expr}")

    # ── 9. Jacobian: implicit function theorem ──
    section("9. IMPLICIT FUNCTION THEOREM DERIVATIVES")
    c = sbm.c_symbol
    P = poly.binding_polynomial
    dP_dc = diff(P, c)
    lines.append(f"  P({c}) = {P}")
    lines.append(f"  dP/d{c} = {dP_dc}")

    for k_sym in sbm.k_symbols:
        dP_dk = diff(P, k_sym)
        dc_dk = simplify(-dP_dk / dP_dc)
        lines.append(f"")
        lines.append(f"  dP/d{k_sym} = {dP_dk}")
        lines.append(f"  d{c}/d{k_sym} = -( dP/d{k_sym} ) / ( dP/d{c} )")
        lines.append(f"              = {dc_dk}")

    # ── 10. Physical Jacobian J_phys ──
    section("10. PHYSICAL JACOBIAN: d[Species]/d[K]")
    lines.append(f"  Using chain rule: dS/dK = (dS/d{c})*(d{c}/dK) + (dS/dK)_partial")
    lines.append(f"  Matrix shape: ({len(poly._micro_species)} species) x ({len(sbm.k_symbols)} K params)")
    lines.append(f"  Rows: {poly._micro_species}")
    lines.append(f"  Cols: {[str(k) for k in sbm.k_symbols]}")

    J = sbm.J_phys_symbolic
    for i, s_name in enumerate(poly._micro_species):
        for j, k_sym in enumerate(sbm.k_symbols):
            entry = J[i, j]
            lines.append(f"")
            lines.append(f"  d[{s_name}]/d{k_sym} = {entry}")

    subsection("J_phys input symbols (for lambdify)")
    lines.append(f"  {[str(s) for s in sbm.input_syms_J_phys]}")

    # ── Write ──
    output_text = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(output_text)

    print(f"Symbolic dump written to: {output_path}")
    print(f"  {len(lines)} lines")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <model_spec_file> [output_file]")
        sys.exit(1)

    spec_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    dump_symbolic(spec_path, output_path)
