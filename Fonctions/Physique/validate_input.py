#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
  pip install jsonschema
  python validate_input.py chemin/vers/ton_fichier.json
"""

import json
import sys
from typing import List, Set, Dict

try:
    from jsonschema import Draft202012Validator
except Exception:
    print("Le package 'jsonschema' est requis. Installe : pip install jsonschema", file=sys.stderr)
    sys.exit(2)

# -------- JSON SCHEMA (intégré) --------
SCHEMA = {
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Workflow Input Schema",
  "type": "object",
  "required": [
    "version","params","shapes","shapeEdgeLines","domainEdits","meshSpecs",
    "materials","materialDomains","domainOwner","matVars",
    "boundarySpecs","boundaryLines","initSpecs","initDomains","solver"
  ],
  "additionalProperties": False,
  "properties": {
    "version": {"type":"integer", "minimum": 1},
    "params": {
      "type":"array",
      "items": {
        "type":"object",
        "required": ["name","unit","value"],
        "additionalProperties": False,
        "properties": {
          "name": {"type":"string", "minLength":1},
          "unit": {"type":"string"},
          "value": {"oneOf":[{"type":"number"},{"type":"string"}]},
          "desc": {"type":"string"}
        }
      }
    },
    "shapes": {
      "type":"array",
      "items": {
        "type":"object",
        "required": ["id","kind"],
        "properties": {
          "id": {"type":"string", "minLength":1},
          "kind": {"type":"string", "enum":["RECT","LINE","CIRCLE","POLY"]},
          "x": {"type":"number"},
          "y": {"type":"number"},
          "w": {"type":"number"},
          "h": {"type":"number"},
          "x1": {"type":"number"},
          "y1": {"type":"number"},
          "x2": {"type":"number"},
          "y2": {"type":"number"},
          "r": {"type":"number"},
          "points": {
            "type":"array",
            "items": {
              "type":"object",
              "required":["x","y"],
              "properties": {"x":{"type":"number"}, "y":{"type":"number"}}
            }
          }
        },
        "additionalProperties": False
      }
    },
    "shapeEdgeLines": {
      "type":"object",
      "additionalProperties": {
        "type":"array",
        "items":{"type":"string"}
      }
    },
    "domainEdits": {
      "type":"object",
      "additionalProperties": {
        "type":"array",
        "items": {
          "type":"object",
          "required":["shapeId","op"],
          "additionalProperties": False,
          "properties": {
            "shapeId": {"type":"string"},
            "op": {"type":"string","enum":["UNION","SUB","INTERSECT"]}
          }
        }
      }
    },
    "meshSpecs": {
      "type":"object",
      "additionalProperties": {
        "type":"object",
        "required":["kind"],
        "additionalProperties": True,
        "properties": {
          "kind":{"type":"string","enum":["TRI_REG","QUAD_REG","TRI_UNSTR","HEX","TET"]},
          "dx":{"type":["number","null"]},
          "dy":{"type":["number","null"]},
          "order":{"type":["integer","null"],"minimum":1,"maximum":3},
          "dxmin":{"type":["number","null"]},
          "dxmax":{"type":["number","null"]},
          "dymin":{"type":["number","null"]},
          "dymax":{"type":["number","null"]},
          "growFromBoundary":{"type":"boolean"}
        }
      }
    },
    "materials": {
      "type":"array",
      "minItems":1,
      "items":{"type":"string"}
    },
    "materialDomains": {
      "type":"object",
      "additionalProperties": {
        "type":"array",
        "items":{"type":"string"}
      }
    },
    "domainOwner": {
      "type":"object",
      "additionalProperties":{"type":"string"}
    },
    "matVars": {
      "type":"object",
      "additionalProperties": {
        "type":"object",
        "additionalProperties": {
          "type":"object",
          "required":["useFunc","funcExpr","csvPath","useT","useP","useConc"],
          "properties": {
            "useFunc":{"type":"boolean"},
            "funcExpr":{"type":"string"},
            "csvPath":{"type":"string"},
            "useT":{"type":"boolean"},
            "useP":{"type":"boolean"},
            "useConc":{"type":"boolean"}
          },
          "additionalProperties": False
        }
      }
    },
    "reactions": {"type":"object"},
    "boundarySpecs": {
      "type":"object",
      "additionalProperties": {
        "type":"object",
        "additionalProperties": {
          "oneOf":[
            {"type":"string"},
            {"type":"number"},
            {
              "type":"object",
              "required":["type"],
              "additionalProperties": True,
              "properties": {
                "type":{"type":"string","enum":["Dirichlet","Neumann","Robin","Slip","NoSlip","WallFunction","Symmetry","Periodic","Outlet","Inlet"]},
                "valueExpr":{"type":"string"},
                "unit":{"type":"string"},
                "frame":{"type":"string","enum":["global","local"]}
              }
            }
          ]
        }
      }
    },
    "boundaryLines": {
      "type":"object",
      "additionalProperties": {
        "type":"array",
        "items":{"type":"string"}
      }
    },
    "initSpecs": {
      "type":"object",
      "additionalProperties": {
        "type":"object",
        "additionalProperties": {
          "oneOf":[{"type":"string"},{"type":"number"}]
        }
      }
    },
    "initDomains": {
      "type":"object",
      "additionalProperties": {
        "type":"array",
        "items":{"type":"string"}
      }
    },
    "physics": {  # optionnel : activation globale déclarative
      "type":"array",
      "items":{"type":"string","enum":["NS","Heat","Species","Maxwell","Porous","MHD","Radiation"]}
    },
    "solver": {
      "type":"object",
      "required":["timeScheme","linSolver","precond","tol","iters"],
      "additionalProperties": True,
      "properties": {
        "timeScheme":{"type":"string","enum":["Euler implicite","Euler explicite","Crank–Nicolson","RK2","RK3","BDF2","IMEX"]},
        "dt":{"type":["number","null"]},
        "tEnd":{"type":["number","null"]},
        "adaptive":{"type":["boolean","null"]},
        "linSolver":{"type":"string","enum":["CG","GMRES","BiCGSTAB","MINRES"]},
        "precond":{"type":"string","enum":["ILU","AMG","Jacobi","None"]},
        "tol":{"oneOf":[{"type":"number"},{"type":"string"}]},
        "iters":{"oneOf":[{"type":"integer"},{"type":"string"}]}
      }
    }
  }
}

# -------- INFÉRENCE DE PHYSIQUES & CONTRÔLES --------

PHYS_KEYS = {
    # needed = variables nécessaires dans matVars pour activer la physique
    # bc_keys = clés de BC qui "demandent" la physique
    "NS":      {"needed": {"rho","mu"},                     "bc_keys": {"ux","uy","uz","p"}},
    "Heat":    {"needed": {"k","cp"},                       "bc_keys": {"T"}},
    "Maxwell": {"needed": {"sigma_e","mu_r","chi_m"},       "bc_keys": {"B","E"}},
    "Species": {"needed": set(),                            "bc_keys": {"mass","species"}},  # activée via 'reactions'
}

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def basic_validate(doc: dict) -> List[str]:
    v = Draft202012Validator(SCHEMA)
    errors = []
    for err in sorted(v.iter_errors(doc), key=str):
        errors.append(f"[SCHEMA] {err.message} @ {'/'.join([str(x) for x in err.path])}")
    return errors

def collect_ids(shapes: List[dict]) -> Set[str]:
    ids, dups = set(), set()
    for s in shapes:
        i = s.get("id")
        if i in ids: dups.add(i)
        ids.add(i)
    if dups:
        raise ValueError(f"Duplicate shape ids found: {sorted(dups)}")
    return ids

def lines_from_shapes(shapes: List[dict]) -> Set[str]:
    return {s["id"] for s in shapes if s.get("kind") == "LINE"}

def infer_material_physics(doc: dict) -> Dict[str, Set[str]]:
    res: Dict[str, Set[str]] = {}
    matvars = doc.get("matVars", {})
    reactions = doc.get("reactions", {})
    reactions_enabled = bool(reactions)

    for mat, props in matvars.items():
        keys = set(props.keys())
        active: Set[str] = set()
        if {"rho","mu"}.issubset(keys):                      # NS
            active.add("NS")
        if {"k","cp"}.issubset(keys):                        # Heat
            active.add("Heat")
        if {"sigma_e","mu_r","chi_m"}.issubset(keys):        # Maxwell
            active.add("Maxwell")
        if reactions_enabled:                                # Species (global si reactions non vides)
            active.add("Species")
        res[mat] = active
    # Matériaux déclarés sans matVars => set vide
    for m in doc.get("materials", []):
        res.setdefault(m, set())
    return res

def infer_domain_physics(doc: dict, mat_phys: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    res: Dict[str, Set[str]] = {}
    for d, m in doc.get("domainOwner", {}).items():
        res[d] = set(mat_phys.get(m, set()))
    return res

def physics_from_bcs(doc: dict) -> Dict[str, Set[str]]:
    b_phys: Dict[str, Set[str]] = {}
    for bname, spec in doc.get("boundarySpecs", {}).items():
        keys = set([k for k in spec.keys() if k != "external"])
        active = set()
        for phys, meta in PHYS_KEYS.items():
            if keys & meta["bc_keys"]:
                active.add(phys)
        b_phys[bname] = active
    return b_phys

def semantic_checks(doc: dict):
    errs: List[str] = []
    shapes = doc.get("shapes", [])
    shape_ids = {s.get("id") for s in shapes}
    try:
        collect_ids(shapes)
    except ValueError as e:
        errs.append(f"[SEMANTIC] {e}")

    line_ids = lines_from_shapes(shapes)
    for sh, lines in doc.get("shapeEdgeLines", {}).items():
        if sh not in shape_ids:
            errs.append(f"[SEMANTIC] shapeEdgeLines references unknown shape '{sh}'")
        for l in lines:
            if l not in line_ids:
                errs.append(f"[SEMANTIC] shapeEdgeLines[{sh}] references unknown line '{l}'")

    for dom, ops in doc.get("domainEdits", {}).items():
        for op in ops:
            sid = op.get("shapeId")
            if sid not in shape_ids:
                errs.append(f"[SEMANTIC] domainEdits[{dom}] uses unknown shapeId '{sid}'")

    for b, lines in doc.get("boundaryLines", {}).items():
        for l in lines:
            if l not in line_ids:
                errs.append(f"[SEMANTIC] boundary '{b}' references unknown line '{l}'")

    allowed_bc_keys = set(["ux","uy","uz","p","T","B","E","mass","species","external"])
    for b, spec in doc.get("boundarySpecs", {}).items():
        for k in spec.keys():
            if k not in allowed_bc_keys:
                errs.append(f"[SEMANTIC] boundarySpecs[{b}] has unknown field '{k}' (pense à un objet BC structuré)")

    for init, doms in doc.get("initDomains", {}).items():
        for d in doms:
            if d not in doc.get("domainOwner", {}):
                errs.append(f"[SEMANTIC] initDomains[{init}] references unknown domain '{d}'")

    materials = set(doc.get("materials", []))
    for m, doms in doc.get("materialDomains", {}).items():
        if m not in materials:
            errs.append(f"[SEMANTIC] materialDomains references unknown material '{m}'")
        for d in doms:
            if d not in doc.get("domainOwner", {}):
                errs.append(f"[SEMANTIC] materialDomains[{m}] references unknown domain '{d}'")

    for d, m in doc.get("domainOwner", {}).items():
        if m not in materials:
            errs.append(f"[SEMANTIC] domainOwner[{d}] references unknown material '{m}'")

    # Params numeric-parse
    for p in doc.get("params", []):
        val = p.get("value")
        if isinstance(val, str):
            try:
                float(val)
            except Exception:
                errs.append(f"[SEMANTIC] params '{p.get('name')}' value '{val}' n'est pas numérique")

    # Solver numeric-parse
    solv = doc.get("solver", {})
    if isinstance(solv.get("tol"), str):
        try:
            float(solv.get("tol"))
        except Exception:
            errs.append("[SEMANTIC] solver.tol doit être un nombre ou une chaîne numérique")

    if isinstance(solv.get("iters"), str):
        try:
            int(solv.get("iters"))
        except Exception:
            errs.append("[SEMANTIC] solver.iters doit être un entier ou une chaîne entière")

    # ---- Physiques : inférence & cohérences ----
    mat_phys = infer_material_physics(doc)
    dom_phys = infer_domain_physics(doc, mat_phys)
    bc_phys = physics_from_bcs(doc)

    # 1) Si 'physics' global existe, doit contenir l'union des physiques inférées par les matériaux
    declared = set(doc.get("physics", []))
    union_inferred = set().union(*dom_phys.values()) if dom_phys else set()
    if declared and not union_inferred.issubset(declared):
        missing = union_inferred - declared
        errs.append(f"[PHYS] 'physics' global manque: {sorted(missing)} (inférés via matériaux)")

    # 2) Frontières demandant des physiques non actives dans les domaines
    all_dom_phys_union = union_inferred
    for b, pset in bc_phys.items():
        extra = pset - all_dom_phys_union
        if extra:
            errs.append(f"[PHYS] frontière '{b}' demande {sorted(extra)} mais aucun domaine ne les active via matériaux")

    # 3) Matériaux partiellement définis
    for m, props in doc.get("matVars", {}).items():
        keys = set(props.keys())
        for tag, meta in PHYS_KEYS.items():
            needed = meta["needed"]
            if needed and (keys & needed) and not needed.issubset(keys):
                missing = needed - keys
                errs.append(f"[PHYS] matériau '{m}' suggère {tag} mais variables manquantes: {sorted(missing)}")

    # 4) Réactions présentes sans champs species/mass en CI/CL
    has_species_field = False
    for b, spec in doc.get("boundarySpecs", {}).items():
        if "mass" in spec or "species" in spec:
            has_species_field = True
            break
    if not has_species_field:
        for ci, spec in doc.get("initSpecs", {}).items():
            if "mass" in spec or "species" in spec:
                has_species_field = True
                break
    if doc.get("reactions", {}) and not has_species_field:
        errs.append("[PHYS] 'reactions' défini mais aucun champ 'mass'/'species' en CI/CL (vérifie la portée)")

    return errs, mat_phys, dom_phys, bc_phys

def print_report(mat_phys, dom_phys, bc_phys):
    def fmt_set(s): return ",".join(sorted(s)) if s else "-"
    print("\n=== Physics Inference Report ===")
    print("Par matériau :")
    for m, s in mat_phys.items():
        print(f" - {m}: {fmt_set(s)}")
    print("Par domaine :")
    for d, s in dom_phys.items():
        print(f" - {d}: {fmt_set(s)}")
    print("Par frontière (demandé par les clés de BC) :")
    for b, s in bc_phys.items():
        print(f" - {b}: {fmt_set(s)}")
    print("================================\n")

def main():
    if len(sys.argv) != 2:
        print("Usage: python validate_input.py <input.json>")
        sys.exit(1)
    path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    schema_errors = basic_validate(doc)
    sem_errors, mat_phys, dom_phys, bc_phys = semantic_checks(doc)
    all_errors = schema_errors + sem_errors

    if all_errors:
        print("Validation FAILED:")
        for e in all_errors:
            print(" -", e)
        print_report(mat_phys, dom_phys, bc_phys)
        sys.exit(1)
    else:
        print("Validation OK.")
        print_report(mat_phys, dom_phys, bc_phys)
        sys.exit(0)

if __name__ == "__main__":
    main()
