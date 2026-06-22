from collections import OrderedDict

# PHYSICAL_PROPERTIES = OrderedDict([
#     ("youngs_modulus",       {"label": "Young’s Modulus(GPa)",          "typical_range": "0.001-1000"}),
#     ("density",              {"label": "Density(g/cm³)",                "typical_range": "0.001-25"}),
#     ("hardness",             {"label": "Hardness Index",                "typical_range": "1-10000"}),
#     ("friction_coefficient", {"label": "Friction Coefficient(μ)",       "typical_range": "0.01-2.0"}),
# ])

# ── Affordances ────────────────────────────────────────────────────────
AFFORDANCES = [
    "grasp", "push", "pull", "lift", "slide", "rotate", "pour", "squeeze", "twist", "press", "hold", "open", "close", "insert", "support", "rest", "attach", "detach", "connect"
]

MATERIAL_PROPERTY_PRIORS = OrderedDict({
        # Metals
        "carbon_steel": {
            "youngs_modulus_GPa": (190.0, 210.0),
            "density_g_cm3": (7.75, 7.90),
            "poissons_ratio": (0.27, 0.30),
            "tensile_strength_MPa": (350.0, 2000.0),
            "hardness": (100.0, 800.0),
            "thermal_conductivity_W_mK": (30.0, 60.0),
        },

        "stainless_steel": {
            "youngs_modulus_GPa": (190.0, 205.0),
            "density_g_cm3": (7.70, 8.10),
            "poissons_ratio": (0.27, 0.31),
            "tensile_strength_MPa": (500.0, 1200.0),
            "hardness": (150.0, 500.0),
            "thermal_conductivity_W_mK": (12.0, 30.0),
        },

        # Aluminium is written as aluminum_alloy because most real products
        # are alloys rather than high-purity aluminium.
        "aluminium": {
            "youngs_modulus_GPa": (68.0, 75.0),
            "density_g_cm3": (2.60, 2.85),
            "poissons_ratio": (0.32, 0.36),
            "tensile_strength_MPa": (70.0, 600.0),
            "hardness": (20.0, 180.0),
            "thermal_conductivity_W_mK": (120.0, 235.0),
        },

        "copper": {
            "youngs_modulus_GPa": (110.0, 130.0),
            "density_g_cm3": (8.80, 9.00),
            "poissons_ratio": (0.33, 0.35),
            "tensile_strength_MPa": (200.0, 400.0),
            "hardness": (40.0, 160.0),
            "thermal_conductivity_W_mK": (350.0, 400.0),
        },

        "brass": {
            "youngs_modulus_GPa": (95.0, 110.0),
            "density_g_cm3": (8.30, 8.70),
            "poissons_ratio": (0.31, 0.34),
            "tensile_strength_MPa": (250.0, 700.0),
            "hardness": (60.0, 250.0),
            "thermal_conductivity_W_mK": (90.0, 160.0),
        },

        "bronze": {
            "youngs_modulus_GPa": (95.0, 120.0),
            "density_g_cm3": (7.40, 8.90),
            "poissons_ratio": (0.30, 0.35),
            "tensile_strength_MPa": (250.0, 900.0),
            "hardness": (60.0, 300.0),
            "thermal_conductivity_W_mK": (40.0, 120.0),
        },

        "titanium": {
            "youngs_modulus_GPa": (100.0, 120.0),
            "density_g_cm3": (4.40, 4.60),
            "poissons_ratio": (0.32, 0.36),
            "tensile_strength_MPa": (240.0, 1400.0),
            "hardness": (120.0, 400.0),
            "thermal_conductivity_W_mK": (6.0, 22.0),
        },

        # Mostly die-cast zinc alloys such as Zamak / ZA alloys.
        "zinc_alloy": {
            "youngs_modulus_GPa": (80.0, 100.0),
            "density_g_cm3": (6.40, 7.20),
            "poissons_ratio": (0.25, 0.30),
            "tensile_strength_MPa": (250.0, 450.0),
            "hardness": (80.0, 150.0),
            "thermal_conductivity_W_mK": (90.0, 125.0),
        },

        # Includes gray cast iron and ductile/nodular cast iron.
        "cast_iron": {
            "youngs_modulus_GPa": (90.0, 170.0),
            "density_g_cm3": (6.90, 7.30),
            "poissons_ratio": (0.21, 0.29),
            "tensile_strength_MPa": (150.0, 900.0),
            "hardness": (150.0, 350.0),
            "thermal_conductivity_W_mK": (25.0, 60.0),
        }
})

NUMERIC_PROPS = list(next(iter(MATERIAL_PROPERTY_PRIORS.values())).keys())

# hardness: Vickers HV 기준으로 통일
# MATERIAL_PROPERTY_PRIORS = OrderedDict({
#     "steel": {
#         "youngs_modulus": (190.0, 210.0),
#         "density": (7.7, 7.9),
#         "hardness": (120.0, 350.0),   # HV
#         "friction_coefficient": (0.3, 0.8),
#     },
#     "stainless_steel": {
#         "youngs_modulus": (190.0, 205.0),
#         "density": (7.7, 8.1),
#         "hardness": (150.0, 300.0),   # HV
#         "friction_coefficient": (0.3, 0.8),
#     },
#     "aluminum": {
#         "youngs_modulus": (68.0, 72.0),
#         "density": (2.6, 2.8),
#         "hardness": (25.0, 180.0),    # HV
#         "friction_coefficient": (0.3, 0.9),
#     },
#     "copper": {
#         "youngs_modulus": (110.0, 130.0),
#         "density": (8.8, 9.0),
#         "hardness": (40.0, 130.0),    # HV
#         "friction_coefficient": (0.3, 0.9),
#     },
#     "brass": {
#         "youngs_modulus": (95.0, 110.0),
#         "density": (8.3, 8.7),
#         "hardness": (60.0, 180.0),    # HV
#         "friction_coefficient": (0.3, 0.8),
#     },

#     "acrylic": {
#         "youngs_modulus": (2.4, 3.3),
#         "density": (1.17, 1.20),
#         "hardness": (15.0, 25.0),     # HV (Rockwell M 85-105 환산)
#         "friction_coefficient": (0.3, 0.7),
#     },
#     "polycarbonate": {
#         "youngs_modulus": (2.0, 2.6),
#         "density": (1.18, 1.22),
#         "hardness": (10.0, 20.0),     # HV (Rockwell M 70-90 환산)
#         "friction_coefficient": (0.3, 0.6),
#     },
#     "polyethylene": {
#         "youngs_modulus": (0.2, 1.5),
#         "density": (0.91, 0.96),
#         "hardness": (3.0, 8.0),       # HV (Rockwell R 25-60 환산)
#         "friction_coefficient": (0.1, 0.4),
#     },
#     "polypropylene": {
#         "youngs_modulus": (1.0, 2.0),
#         "density": (0.89, 0.92),
#         "hardness": (5.0, 12.0),      # HV (Rockwell R 50-100 환산)
#         "friction_coefficient": (0.2, 0.5),
#     },
#     "pvc": {
#         "youngs_modulus": (2.0, 4.0),
#         "density": (1.30, 1.45),
#         "hardness": (10.0, 20.0),     # HV (Rockwell R 110-120 환산)
#         "friction_coefficient": (0.3, 0.6),
#     },
#     "nylon": {
#         "youngs_modulus": (1.5, 3.5),
#         "density": (1.10, 1.20),
#         "hardness": (8.0, 20.0),      # HV (Rockwell R 100-120 환산)
#         "friction_coefficient": (0.2, 0.5),
#     },

#     "rubber": {
#         "youngs_modulus": (0.005, 0.10),
#         "density": (0.9, 1.3),
#         "hardness": (0.5, 5.0),       # HV (Shore A 30-90 환산)
#         "friction_coefficient": (0.6, 1.8),
#     },
#     "silicone": {
#         "youngs_modulus": (0.005, 0.05),
#         "density": (1.0, 1.3),
#         "hardness": (0.1, 3.0),       # HV (Shore A 10-70 환산)
#         "friction_coefficient": (0.5, 1.5),
#     },
#     "foam": {
#         "youngs_modulus": (0.001, 0.10),
#         "density": (0.02, 0.3),
#         "hardness": (0.05, 0.5),      # HV (Shore A < 20, 극연질)
#         "friction_coefficient": (0.4, 1.2),
#     },

#     "glass": {
#         "youngs_modulus": (50.0, 90.0),
#         "density": (2.2, 2.8),
#         "hardness": (450.0, 700.0),   # HV
#         "friction_coefficient": (0.4, 0.9),
#     },
#     "ceramic": {
#         "youngs_modulus": (150.0, 400.0),
#         "density": (2.5, 6.0),
#         "hardness": (800.0, 2500.0),  # HV
#         "friction_coefficient": (0.2, 0.8),
#     },
#     "concrete": {
#         "youngs_modulus": (15.0, 40.0),
#         "density": (2.0, 2.5),
#         "hardness": (30.0, 80.0),     # HV (기존 50-150은 과대평가)
#         "friction_coefficient": (0.5, 1.0),
#     },
#     "stone": {
#         "youngs_modulus": (20.0, 100.0),
#         "density": (2.3, 3.0),
#         "hardness": (100.0, 800.0),   # HV (석회암~화강암 범위)
#         "friction_coefficient": (0.4, 1.0),
#     },

#     "wood": {
#         "youngs_modulus": (6.0, 20.0),
#         "density": (0.3, 1.0),
#         "hardness": (10.0, 80.0),     # HV (Janka 환산 근사)
#         "friction_coefficient": (0.3, 0.8),
#     },
#     "plywood": {
#         "youngs_modulus": (4.0, 12.0),
#         "density": (0.4, 0.8),
#         "hardness": (10.0, 60.0),     # HV
#         "friction_coefficient": (0.3, 0.8),
#     },
#     "fabric": {
#         "youngs_modulus": (0.01, 2.0),
#         "density": (0.1, 1.5),
#         "hardness": (0.5, 3.0),       # HV (극연질 섬유 기준)
#         "friction_coefficient": (0.4, 1.2),
#     },
#     "leather": {
#         "youngs_modulus": (0.05, 0.5),
#         "density": (0.8, 1.1),
#         "hardness": (2.0, 10.0),      # HV
#         "friction_coefficient": (0.4, 1.0),
#     },
#     "paper_cardboard": {
#         "youngs_modulus": (0.5, 5.0),
#         "density": (0.5, 1.0),
#         "hardness": (1.0, 10.0),      # HV
#         "friction_coefficient": (0.3, 0.8),
#     },

#     "carbon_fiber": {
#         "youngs_modulus": (70.0, 250.0),
#         "density": (1.5, 1.8),
#         "hardness": (200.0, 600.0),   # HV (CFRP 복합재 기준)
#         "friction_coefficient": (0.2, 0.6),
#     },
# })
