from collections import OrderedDict

# Affordances 
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
            "hardness_HV": (100.0, 800.0),
            "thermal_conductivity_W_mK": (30.0, 60.0),
            "electrical_conductivity_MS_m": (3.0, 10.0),
            "thermal_expansion_coeff_1e-6_K": (10.0, 13.0),
            "fracture_toughness_MPa_sqrt_m": (20.0, 150.0),
        },

        "stainless_steel": {
            "youngs_modulus_GPa": (190.0, 205.0),
            "density_g_cm3": (7.70, 8.10),
            "poissons_ratio": (0.27, 0.31),
            "tensile_strength_MPa": (500.0, 1200.0),
            "hardness_HV": (150.0, 500.0),
            "thermal_conductivity_W_mK": (12.0, 30.0),
            "electrical_conductivity_MS_m": (1.2, 2.0),
            "thermal_expansion_coeff_1e-6_K": (10.0, 18.0),
            "fracture_toughness_MPa_sqrt_m": (40.0, 220.0),
        },

        "aluminium": {
            "youngs_modulus_GPa": (68.0, 75.0),
            "density_g_cm3": (2.60, 2.85),
            "poissons_ratio": (0.32, 0.36),
            "tensile_strength_MPa": (70.0, 600.0),
            "hardness_HV": (20.0, 180.0),
            "thermal_conductivity_W_mK": (120.0, 235.0),
            "electrical_conductivity_MS_m": (18.0, 38.0),
            "thermal_expansion_coeff_1e-6_K": (21.0, 24.0),
            "fracture_toughness_MPa_sqrt_m": (15.0, 45.0),
        },

        "titanium": {
            "youngs_modulus_GPa": (100.0, 120.0),
            "density_g_cm3": (4.40, 4.60),
            "poissons_ratio": (0.32, 0.36),
            "tensile_strength_MPa": (240.0, 1400.0),
            "hardness_HV": (120.0, 400.0),
            "thermal_conductivity_W_mK": (6.0, 22.0),
            "electrical_conductivity_MS_m": (0.5, 2.5),
            "thermal_expansion_coeff_1e-6_K": (8.0, 10.0),
            "fracture_toughness_MPa_sqrt_m": (30.0, 120.0),
        },
        
        "copper": {
            "youngs_modulus_GPa": (110.0, 130.0),
            "density_g_cm3": (8.80, 9.00),
            "poissons_ratio": (0.33, 0.35),
            "tensile_strength_MPa": (200.0, 400.0),
            "hardness_HV": (40.0, 160.0),
            "thermal_conductivity_W_mK": (350.0, 400.0),
            "electrical_conductivity_MS_m": (55.0, 59.0),
            "thermal_expansion_coeff_1e-6_K": (16.0, 18.0),
            "fracture_toughness_MPa_sqrt_m": (30.0, 120.0),
        },

        "brass": {
            "youngs_modulus_GPa": (95.0, 110.0),
            "density_g_cm3": (8.30, 8.70),
            "poissons_ratio": (0.31, 0.34),
            "tensile_strength_MPa": (250.0, 700.0),
            "hardness_HV": (60.0, 250.0),
            "thermal_conductivity_W_mK": (90.0, 160.0),
            "electrical_conductivity_MS_m": (10.0, 18.0),
            "thermal_expansion_coeff_1e-6_K": (18.0, 21.0),
            "fracture_toughness_MPa_sqrt_m": (25.0, 70.0),
        },

        "bronze": {
            "youngs_modulus_GPa": (95.0, 120.0),
            "density_g_cm3": (7.40, 8.90),
            "poissons_ratio": (0.30, 0.35),
            "tensile_strength_MPa": (250.0, 900.0),
            "hardness_HV": (60.0, 300.0),
            "thermal_conductivity_W_mK": (35.0, 85.0),
            "electrical_conductivity_MS_m": (5.0, 15.0),
            "thermal_expansion_coeff_1e-6_K": (16.0, 19.0),
            "fracture_toughness_MPa_sqrt_m": (24.0, 90.0),
        },

        "cast_iron": {
            "youngs_modulus_GPa": (90.0, 170.0),
            "density_g_cm3": (6.90, 7.30),
            "poissons_ratio": (0.21, 0.29),
            "tensile_strength_MPa": (150.0, 900.0),
            "hardness_HV": (150.0, 350.0),
            "thermal_conductivity_W_mK": (25.0, 60.0),
            "electrical_conductivity_MS_m": (0.8, 1.6),
            "thermal_expansion_coeff_1e-6_K": (9.0, 12.0),
            "fracture_toughness_MPa_sqrt_m": (10.0, 60.0),
        }
})

NUMERIC_PROPS = list(next(iter(MATERIAL_PROPERTY_PRIORS.values())).keys())

