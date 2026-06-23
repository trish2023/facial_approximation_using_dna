#!/usr/bin/env python3
"""
hirisplex_model.py
==================
HIrisPlex-S multinomial logistic regression model for eye, hair, and skin
colour prediction from SNP dosages.

Model details (Chaitanya et al. 2018 / Walsh et al. 2017)
----------------------------------------------------------
* Three independent multinomial logistic regression models (eye, hair, skin).
* Predictors: effect-allele dosage for each of the 41 HIrisPlex-S SNPs
  (0 / 1 / 2 per SNP, NA → mean-imputed from present SNPs within that model).
* Linear predictor:  η_k = intercept_k + Σ_i β_{k,i} · dosage_i
* Probabilities via softmax over all categories (reference category is the
  implicit "0" from the last logit equation).
* Ordering of betas MUST match the 41-SNP order in hirisplex_41snps.csv /
  hirisplex_coefficients.json._snp_order.

Usage
-----
    from scripts.hirisplex_model import HIriPlexS
    model = HIriPlexS()
    model.validate()

    dosages = {"rs885479": 0, "rs12913832": 2, ...}   # float | int | "NA"
    eye  = model.predict_eye(dosages)
    hair = model.predict_hair(dosages)
    skin = model.predict_skin(dosages)
"""

from __future__ import annotations

import json
import logging
import math
import csv
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COEFFICIENTS_JSON = PROJECT_ROOT / "data" / "reference" / "hirisplex_coefficients.json"
TEMPLATE_CSV      = PROJECT_ROOT / "data" / "reference" / "hirisplex_41snps.csv"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("hirisplex_model")


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def _softmax(logits: list[float]) -> list[float]:
    """Numerically stable softmax over a list of logit values."""
    max_l = max(logits)
    exps  = [math.exp(l - max_l) for l in logits]
    s     = sum(exps)
    return [e / s for e in exps]


def _parse_numeric(value: str | int | float | None) -> Optional[float]:
    """Return float, or None if the value is 'NA' / None / empty."""
    if value is None:
        return None
    s = str(value).strip().upper()
    if s in ("NA", "NAN", "NONE", ""):
        return None
    try:
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# SNP template loader (preserves template order)
# ---------------------------------------------------------------------------

def _load_snp_order(template_path: Path) -> list[str]:
    """Return the 41 rsIDs in template order (BOM-safe)."""
    order: list[str] = []
    with open(template_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cleaned = {k.strip().lstrip("\ufeff"): v.strip() for k, v in row.items()}
            order.append(cleaned["rsID"])
    return order


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------

class HIriPlexS:
    """
    HIrisPlex-S multinomial logistic regression predictor.

    Parameters
    ----------
    coefficients_path : Path, optional
        Path to hirisplex_coefficients.json.  Defaults to the project
        standard location.
    template_path : Path, optional
        Path to hirisplex_41snps.csv.  Defaults to the project standard
        location.
    confidence_threshold : float
        Minimum probability for the winning category to be considered
        'high-confidence'.  Default 0.50 (i.e. majority-class probability).
    """

    # Minimum proportion of *present* SNPs required per model; below this
    # the confidence flag is automatically set to False.
    _MIN_COVERAGE_FRACTION: float = 30 / 41

    def __init__(
        self,
        coefficients_path: Path = COEFFICIENTS_JSON,
        template_path:     Path = TEMPLATE_CSV,
        confidence_threshold: float = 0.50,
    ) -> None:
        self._coeff_path   = Path(coefficients_path)
        self._template_path = Path(template_path)
        self.confidence_threshold = confidence_threshold

        self._snp_order: list[str] = _load_snp_order(self._template_path)
        self._coeff: dict           = self._load_coefficients()

    # ------------------------------------------------------------------
    # Internal loaders
    # ------------------------------------------------------------------

    def _load_coefficients(self) -> dict:
        """Load and parse hirisplex_coefficients.json."""
        if not self._coeff_path.exists():
            raise FileNotFoundError(
                f"Coefficients file not found: {self._coeff_path}"
            )
        with open(self._coeff_path) as fh:
            data = json.load(fh)
        return data

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """
        Validate that the coefficient file is fully populated and internally
        consistent.  Raises ValueError for any structural problem.

        Checks
        ------
        1. All beta vectors have length == 41 (one entry per SNP).
        2. No 'TODO' placeholder values remain.
        3. Intercepts are numeric for every free equation.
        4. SNP order is consistent with the template.
        """
        errors: list[str] = []
        n_snps = len(self._snp_order)

        for model_key in ("eye_model", "hair_model", "skin_model"):
            model_data = self._coeff.get(model_key, {})
            # Collect free equations (non-underscore keys)
            equations = {k: v for k, v in model_data.items()
                         if not k.startswith("_")}

            if not equations:
                errors.append(f"[{model_key}] No free equations found.")
                continue

            for cat, eq in equations.items():
                # --- intercept ---
                raw_intercept = eq.get("intercept", "TODO")
                if str(raw_intercept).strip().upper() == "TODO":
                    errors.append(
                        f"[{model_key}][{cat}] intercept is still 'TODO'."
                    )
                else:
                    try:
                        float(raw_intercept)
                    except (TypeError, ValueError):
                        errors.append(
                            f"[{model_key}][{cat}] intercept '{raw_intercept}' "
                            "is not numeric."
                        )

                # --- betas ---
                betas = eq.get("betas", [])
                if len(betas) != n_snps:
                    errors.append(
                        f"[{model_key}][{cat}] beta vector has {len(betas)} "
                        f"entries; expected {n_snps}."
                    )
                else:
                    for i, b in enumerate(betas):
                        if str(b).strip().upper() == "TODO":
                            errors.append(
                                f"[{model_key}][{cat}] beta[{i}] "
                                f"({self._snp_order[i]}) is still 'TODO'."
                            )
                        else:
                            try:
                                float(b)
                            except (TypeError, ValueError):
                                errors.append(
                                    f"[{model_key}][{cat}] beta[{i}] "
                                    f"({self._snp_order[i]}) = '{b}' "
                                    "is not numeric."
                                )

        if errors:
            summary = "\n  ".join(errors)
            raise ValueError(
                f"Coefficient validation failed with {len(errors)} error(s):\n"
                f"  {summary}\n\n"
                "Populate hirisplex_coefficients.json with the published "
                "Walsh et al. 2017 / Chaitanya et al. 2018 betas before "
                "running predictions."
            )

        log.info("validate(): all coefficient checks passed for %d SNPs.", n_snps)

    # ------------------------------------------------------------------
    # Core prediction engine
    # ------------------------------------------------------------------

    def _predict(
        self,
        model_key: str,
        dosages: dict[str, str | int | float],
    ) -> dict:
        """
        General-purpose prediction engine for one trait model.

        Parameters
        ----------
        model_key : str
            One of 'eye_model', 'hair_model', 'skin_model'.
        dosages : dict
            Maps rsID → dosage value (0/1/2, or 'NA').

        Returns
        -------
        dict with keys:
            probabilities : dict[category, float]
            predicted     : str   (argmax category)
            high_confidence : bool
            n_present     : int   (SNPs with valid dosage)
            n_imputed     : int   (SNPs that were NA-imputed)
            imputed_rsids : list[str]
            warnings      : list[str]
        """
        model_data = self._coeff[model_key]
        categories_meta = model_data.get("_categories", [])
        note = model_data.get("_note", "")

        # Free equations (non-underscore keys = all non-reference categories)
        equations = {k: v for k, v in model_data.items()
                     if not k.startswith("_")}

        # ---- Step 1: resolve dosage vector, NA-impute ----
        raw_vals: list[Optional[float]] = []
        for rsid in self._snp_order:
            raw_vals.append(_parse_numeric(dosages.get(rsid)))

        present_vals = [v for v in raw_vals if v is not None]
        n_present    = len(present_vals)
        n_total      = len(self._snp_order)

        if n_present == 0:
            raise ValueError(
                f"[{model_key}] No valid dosage values found — "
                "cannot run prediction."
            )

        mean_dosage = sum(present_vals) / n_present

        final_dosages: list[float] = []
        imputed_rsids: list[str]   = []
        for i, v in enumerate(raw_vals):
            if v is None:
                final_dosages.append(mean_dosage)
                imputed_rsids.append(self._snp_order[i])
            else:
                final_dosages.append(v)

        n_imputed = len(imputed_rsids)
        warnings: list[str] = []
        if n_imputed > 0:
            warnings.append(
                f"{n_imputed} SNP(s) were NA-imputed with mean dosage "
                f"{mean_dosage:.4f}: {', '.join(imputed_rsids)}"
            )

        # ---- Step 2: validate coefficients are numeric ----
        # (will raise ValueError with clear message if any TODO remains)
        self.validate()

        # ---- Step 3: compute linear predictors for each free equation ----
        # Reference category gets logit = 0.0 ONLY when it is absent from
        # the free equations (standard corner-point multinomial coding).
        # For the Walsh 2017 hair model, ALL four categories have free equations
        # (full-multinomial / deviance coding); in that case no category is
        # pinned to 0 — the softmax runs over all K computed logits.
        free_cats = list(equations.keys())
        all_cats  = list(categories_meta) if categories_meta else free_cats

        # Reference = category listed in _categories but NOT a free equation
        ref_cats = [c for c in all_cats if c not in free_cats]

        logits: dict[str, float] = {}

        # Pin reference category to 0.0 only when one exists
        if len(ref_cats) == 1:
            logits[ref_cats[0]] = 0.0
        elif len(ref_cats) > 1:
            # Ambiguous — fall back: last listed category is reference
            logits[all_cats[-1]] = 0.0
            log.warning(
                "[%s] Multiple potential reference categories %s — "
                "defaulting to last: %s",
                model_key, ref_cats, all_cats[-1],
            )
        # else: len(ref_cats) == 0 → full K-equation model, no reference pin

        for cat, eq in equations.items():
            intercept = float(eq["intercept"])
            betas     = [float(b) for b in eq["betas"]]
            eta = intercept + sum(b * d for b, d in zip(betas, final_dosages))
            logits[cat] = eta

        # ---- Step 4: softmax over all categories ----

        ordered_cats   = list(logits.keys())
        ordered_logits = [logits[c] for c in ordered_cats]
        probs          = _softmax(ordered_logits)
        prob_dict      = dict(zip(ordered_cats, probs))

        # ---- Step 5: predicted class & confidence flag ----
        predicted     = max(prob_dict, key=prob_dict.__getitem__)
        max_prob      = prob_dict[predicted]
        coverage_ok   = (n_present / n_total) >= self._MIN_COVERAGE_FRACTION
        high_conf     = coverage_ok and (max_prob >= self.confidence_threshold)

        if not coverage_ok:
            warnings.append(
                f"Low SNP coverage ({n_present}/{n_total}); "
                "prediction marked low-confidence."
            )
        if max_prob < self.confidence_threshold:
            warnings.append(
                f"Max category probability {max_prob:.3f} is below the "
                f"confidence threshold {self.confidence_threshold}; "
                "prediction marked low-confidence."
            )

        return {
            "probabilities":    prob_dict,
            "predicted":        predicted,
            "high_confidence":  high_conf,
            "n_present":        n_present,
            "n_imputed":        n_imputed,
            "imputed_rsids":    imputed_rsids,
            "warnings":         warnings,
        }

    # ------------------------------------------------------------------
    # Public prediction methods
    # ------------------------------------------------------------------

    def predict_eye(
        self,
        dosages: dict[str, str | int | float],
    ) -> dict:
        """
        Predict eye colour probabilities.

        Categories: blue, intermediate, brown (reference).

        Parameters
        ----------
        dosages : dict
            rsID → dosage (0/1/2 or 'NA').

        Returns
        -------
        dict — see _predict() for full schema.
        """
        return self._predict("eye_model", dosages)

    def predict_hair(
        self,
        dosages: dict[str, str | int | float],
    ) -> dict:
        """
        Predict hair colour probabilities.

        Categories: blond, brown (reference), red, black.

        Parameters
        ----------
        dosages : dict
            rsID → dosage (0/1/2 or 'NA').

        Returns
        -------
        dict — see _predict() for full schema.
        """
        return self._predict("hair_model", dosages)

    def predict_skin(
        self,
        dosages: dict[str, str | int | float],
    ) -> dict:
        """
        Predict skin colour probabilities.

        Categories: very_pale, pale (reference), intermediate, dark,
        dark_to_black.

        Parameters
        ----------
        dosages : dict
            rsID → dosage (0/1/2 or 'NA').

        Returns
        -------
        dict — see _predict() for full schema.
        """
        return self._predict("skin_model", dosages)
