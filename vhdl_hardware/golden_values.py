"""
Golden Values for VHDL Hardware Simulation
==========================================

This file contains the original (golden) values for FMAP filters and BIAS values
extracted from the VHDL file CONV1_SAB_STUCK_DECOS.vhd

These values represent the unmodified, reference weights and biases that should
be used for golden simulation (baseline simulation without fault injection).
"""

# Original FMAP filter values (6 filters, each 5x5)
GOLDEN_FMAP_VALUES = {
    "FMAP_1": [
        ["11101010", "11110001", "11101101", "11110111", "00001111"],
        ["00000101", "00000111", "00001100", "00000110", "00001001"],
        ["11101110", "00001000", "00011010", "00001010", "00000101"],
        ["00000111", "00011000", "00011000", "00000010", "11110101"],
        ["11111011", "00000101", "00001010", "11110100", "11101011"]
    ],
    "FMAP_2": [
        ["00001110", "00010100", "00010011", "00000101", "00011111"],
        ["00011011", "00001001", "00010101", "00011001", "00000000"],
        ["11111111", "00001010", "00001101", "11110011", "11110101"],
        ["11100001", "11100110", "11101110", "11111001", "11111010"],
        ["11100100", "11101101", "11010110", "11100110", "00000000"]
    ],
    "FMAP_3": [
        ["11110010", "11101100", "11111111", "00100001", "00101000"],
        ["11101000", "11110110", "11110110", "00001011", "00011100"],
        ["11100110", "11011001", "00000000", "11111101", "00001111"],
        ["11101010", "11110100", "00000010", "00000001", "00001111"],
        ["11111001", "11101101", "11101011", "11101101", "00001100"]
    ],
    "FMAP_4": [
        ["11101000", "00001100", "11110101", "00000100", "00010101"],
        ["11110010", "11101011", "00010001", "00011011", "11111001"],
        ["11101011", "11110000", "00010000", "00001100", "00010010"],
        ["11111011", "00010101", "00000111", "00000100", "00000000"],
        ["11111011", "00000111", "00010000", "00000111", "11110010"]
    ],
    "FMAP_5": [
        ["00001111", "00000101", "00001111", "11101011", "11111001"],
        ["00011111", "00100000", "00001001", "11100100", "11100000"],
        ["00010000", "00010101", "11110110", "11110000", "11011000"],
        ["00001000", "00000000", "00001110", "00000111", "00000110"],
        ["00000100", "00001111", "00000101", "11110100", "00001011"]
    ],
    "FMAP_6": [
        ["11111001", "11100111", "11101000", "11011000", "11011011"],
        ["00000110", "11111101", "11101011", "11100011", "00000000"],
        ["00011111", "00000010", "00010100", "00001101", "00000111"],
        ["00100000", "00000111", "00100010", "11111110", "11111111"],
        ["00001001", "00010100", "00001010", "00011010", "11111110"]
    ]
}

# Original BIAS values (6 bias values, each 16-bit signed)
GOLDEN_BIAS_VALUES = {
    "BIAS_VAL_1": "1111111111111110",
    "BIAS_VAL_2": "0000000000001001",
    "BIAS_VAL_3": "0000000000001001",
    "BIAS_VAL_4": "1111111111110101",
    "BIAS_VAL_5": "1111111111110111",
    "BIAS_VAL_6": "1111111111111100"
}

def get_golden_fmap_values():
    """
    Returns the golden FMAP filter values.
    
    Returns:
        dict: Dictionary containing all 6 FMAP filters with their original values
    """
    return GOLDEN_FMAP_VALUES.copy()

def get_golden_bias_values():
    """
    Returns the golden BIAS values.
    
    Returns:
        dict: Dictionary containing all 6 BIAS values with their original values
    """
    return GOLDEN_BIAS_VALUES.copy()

def get_all_golden_values():
    """
    Returns both FMAP and BIAS golden values.
    
    Returns:
        dict: Dictionary containing both 'fmap' and 'bias' golden values
    """
    return {
        "fmap": get_golden_fmap_values(),
        "bias": get_golden_bias_values()
    }

def format_fmap_for_vhdl(fmap_name, fmap_values):
    """
    Formats FMAP values for VHDL constant declaration.
    
    Args:
        fmap_name (str): Name of the FMAP (e.g., "FMAP_1")
        fmap_values (list): 5x5 matrix of binary string values
        
    Returns:
        str: Formatted VHDL constant declaration
    """
    vhdl_lines = [f"\tconstant {fmap_name}: FILTER_TYPE:= ("]
    
    for i, row in enumerate(fmap_values):
        row_str = '(' + ','.join([f'"{val}"' for val in row]) + ')'
        if i < len(fmap_values) - 1:
            row_str += ','
        vhdl_lines.append(f"\t\t{row_str}")
    
    vhdl_lines.append("\t);")
    return '\n'.join(vhdl_lines)

def format_bias_for_vhdl(bias_name, bias_value):
    """
    Formats BIAS value for VHDL constant declaration.
    
    Args:
        bias_name (str): Name of the BIAS (e.g., "BIAS_VAL_1")
        bias_value (str): 16-bit binary string value
        
    Returns:
        str: Formatted VHDL constant declaration
    """
    return f'\tconstant {bias_name}: signed (15 downto 0) := "{bias_value}";'

def generate_vhdl_constants():
    """
    Generates complete VHDL constant declarations for all golden values.
    
    Returns:
        str: Complete VHDL constant declarations ready to be inserted into VHDL file
    """
    vhdl_content = []
    
    # Add FMAP constants
    for fmap_name, fmap_values in GOLDEN_FMAP_VALUES.items():
        vhdl_content.append(format_fmap_for_vhdl(fmap_name, fmap_values))
        vhdl_content.append("")  # Empty line between constants
    
    # Add BIAS constants
    for bias_name, bias_value in GOLDEN_BIAS_VALUES.items():
        vhdl_content.append(format_bias_for_vhdl(bias_name, bias_value))
    
    return '\n'.join(vhdl_content)