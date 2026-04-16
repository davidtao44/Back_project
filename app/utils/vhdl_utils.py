import re


def to_binary_c2(value, bits=8):
    """Convierte un valor a binario en complemento a 2."""
    if value < 0:
        value = (1 << bits) + value
    return format(value, f"0{bits}b")


def apply_bit_faults(binary_string: str, bit_positions: list, fault_type: str = "bitflip") -> str:
    """Aplica fallos de bit según el tipo especificado a una cadena binaria."""
    if not bit_positions:
        return binary_string

    bits = list(binary_string)

    for bit_pos in bit_positions:
        if 0 <= bit_pos < len(bits):
            original_bit = bits[bit_pos]

            if fault_type == "stuck_at_0":
                bits[bit_pos] = "0"
                print(f"  🔒 Bit {bit_pos}: {original_bit} → 0 (stuck-at-0)")
            elif fault_type == "stuck_at_1":
                bits[bit_pos] = "1"
                print(f"  🔒 Bit {bit_pos}: {original_bit} → 1 (stuck-at-1)")
            else:
                bits[bit_pos] = "1" if bits[bit_pos] == "0" else "0"
                print(f"  🔄 Bit {bit_pos}: {original_bit} → {bits[bit_pos]} (bit-flip)")
        else:
            print(f"  ⚠️ Posición de bit inválida: {bit_pos} (longitud: {len(bits)})")

    return "".join(bits)


def generate_vhdl_code(vhdl_matrix, width, height):
    """Genera código VHDL con la estructura de Memoria_Imagen.vhd."""
    vhdl_code = [
        "library ieee; ",
        "use ieee.std_logic_1164.all; ",
        "use ieee.numeric_std.all; ",
        "",
        "entity Memoria_Imagen is",
        "",
        "\tgeneric (",
        "\t\tdata_width  : natural := 8; ",
        "\t   addr_length : natural := 10\t-- 1024 pos mem",
        "\t\t); ",
        "\t",
        "\tport ( ",
        "\t\tclk      :  in std_logic;",
        "--\t\trst : in std_logic;",
        "\t\taddress  :  in std_logic_vector(addr_length-1 downto 0); ",
        "\t\tdata_out :  out std_logic_vector(data_width-1  downto 0) ",
        "\t\t);",
        "\t\t",
        "end Memoria_Imagen;",
        "",
        "",
        "architecture synth of Memoria_Imagen is",
        "",
        "\tconstant mem_size : natural := 2**addr_length; \t",
        "\ttype mem_type is array (0 to mem_size-1) of std_logic_vector (data_width-1 downto 0); ",
        "",
        "constant mem : mem_type := (",
    ]

    flat_matrix = []
    for row in vhdl_matrix:
        flat_matrix.extend(row)

    for i in range(0, len(flat_matrix), 32):
        chunk = flat_matrix[i:i + 32]
        line = "\t\t" + ", ".join(chunk)
        if i + 32 < len(flat_matrix):
            line += ","
        vhdl_code.append(line)

    vhdl_code.extend([
        "\t);",
        "",
        "\t",
        "begin ",
        "",
        "\trom : process (clk) ",
        "\tbegin",
        "\t   ",
        "----\t   if (rising_edge(Clk)) then",
        "--\t\t  if rst = '1' then",
        "--\t\t\t\t-- Reset the counter to 0",
        "--\t\t\t\tdata_out <= ((others=> '0'));",
        "--\t\t  end if;",
        "\t   ",
        "\t\tif rising_edge(clk) then ",
        "\t\t\tdata_out <= mem(to_integer(unsigned(address))); ",
        "\t\tend if; ",
        "\tend process rom; ",
        "",
        "end architecture synth;",
    ])

    return "\n".join(vhdl_code)


def modify_filter_in_vhdl(content: str, filter_name: str, positions: list) -> str:
    """Modifica un filtro específico (FMAP_X) en el contenido VHDL."""
    pattern = rf"constant {filter_name}: FILTER_TYPE:=\s*\((.*?)\);"
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        print(f"⚠️ No se encontró el filtro {filter_name}")
        return content

    filter_definition = match.group(1)
    print(f"✅ Encontrado filtro {filter_name}")

    rows = []
    for line in filter_definition.split("\n"):
        line = line.strip()
        if line.startswith("(") and line.endswith("),") or line.endswith(")"):
            row_match = re.search(r"\((.*?)\)", line)
            if row_match:
                values = [v.strip().strip('"') for v in row_match.group(1).split(",")]
                rows.append(values)

    print(f"🔍 Matriz original del filtro {filter_name}: {len(rows)}x{len(rows[0]) if rows else 0}")

    for pos_config in positions:
        row = pos_config.get("row", 0)
        col = pos_config.get("col", 0)
        bit_positions = pos_config.get("bit_positions", [])
        fault_type = pos_config.get("fault_type", "bitflip")

        if 0 <= row < len(rows) and 0 <= col < len(rows[row]):
            original_value = rows[row][col]
            modified_value = apply_bit_faults(original_value, bit_positions, fault_type)
            rows[row][col] = modified_value
            print(f"✅ Modificado {filter_name}[{row}][{col}]: {original_value} → {modified_value} (tipo: {fault_type})")
        else:
            print(f"⚠️ Posición inválida {filter_name}[{row}][{col}]")

    new_filter_lines = []
    for i, row in enumerate(rows):
        formatted_values = ",".join([f'"{val}"' for val in row])
        if i == len(rows) - 1:
            new_filter_lines.append(f"\t\t({formatted_values})")
        else:
            new_filter_lines.append(f"\t\t({formatted_values}),")

    new_filter_definition = (
        f"constant {filter_name}: FILTER_TYPE:= (\n" + "\n".join(new_filter_lines) + "\n\t);"
    )

    return content.replace(match.group(0), new_filter_definition)


def modify_bias_in_vhdl(content: str, bias_name: str, fault_infos: list) -> str:
    """Modifica un bias específico (BIAS_VAL_X) en el contenido VHDL."""
    pattern = rf'constant {bias_name}: signed \(15 downto 0\) := "([01]+)";'
    match = re.search(pattern, content)

    if not match:
        print(f"⚠️ No se encontró el bias {bias_name}")
        return content

    original_value = match.group(1)
    modified_value = original_value

    for fault_info in fault_infos:
        bit_position = fault_info.get("bit_position", 0)
        fault_type = fault_info.get("fault_type", "bitflip")
        modified_value = apply_bit_faults(modified_value, [bit_position], fault_type)

    print(f"✅ Modificado {bias_name}: {original_value} → {modified_value}")

    new_definition = f'constant {bias_name}: signed (15 downto 0) := "{modified_value}";'
    return content.replace(match.group(0), new_definition)


def modify_vhdl_weights_and_bias(content: str, fault_config: dict) -> str:
    """Modifica los pesos (FMAP_X) y bias (BIAS_VAL_X) en el contenido del archivo VHDL."""
    modified_content = content

    filter_faults = fault_config.get("filter_faults", [])
    if filter_faults:
        print(f"🔧 Modificando filtros: {filter_faults}")

        for fault in filter_faults:
            filter_name = fault.get("filter_name", "")
            if filter_name:
                position_config = {
                    "row": int(fault.get("row", 0)),
                    "col": int(fault.get("col", 0)),
                    "bit_positions": [int(fault.get("bit_position", 0))],
                    "fault_type": fault.get("fault_type", "bitflip"),
                }

                print(
                    f"🔧 Procesando {filter_name} en posición [{position_config['row']}][{position_config['col']}] "
                    f"bit {position_config['bit_positions'][0]} con tipo: {position_config['fault_type']}"
                )
                modified_content = modify_filter_in_vhdl(modified_content, filter_name, [position_config])

    bias_faults = fault_config.get("bias_faults", [])
    if bias_faults:
        print(f"🔧 Modificando bias: {bias_faults}")

        bias_groups = {}
        for fault in bias_faults:
            bias_name = fault.get("bias_name", "")
            if bias_name not in bias_groups:
                bias_groups[bias_name] = []

            bias_groups[bias_name].append(
                {
                    "bit_position": int(fault.get("bit_position", 0)),
                    "fault_type": fault.get("fault_type", "bitflip"),
                }
            )

        for bias_name, fault_infos in bias_groups.items():
            if fault_infos:
                print(f"🔧 Procesando {bias_name} con fallos: {fault_infos}")
                modified_content = modify_bias_in_vhdl(modified_content, bias_name, fault_infos)

    return modified_content


def replace_filter_with_golden_values(content: str, filter_name: str, golden_matrix: list) -> str:
    """Reemplaza completamente un filtro FMAP con los valores golden originales."""
    pattern = rf"constant {filter_name}: FILTER_TYPE:=\s*\((.*?)\);"
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        print(f"⚠️ No se encontró el filtro {filter_name}")
        return content

    print(f"✅ Encontrado filtro {filter_name}")
    print(f"🔍 Matriz original del filtro {filter_name}: 5x5")

    new_filter_lines = []
    for i, row in enumerate(golden_matrix):
        formatted_values = ",".join([f'"{val}"' for val in row])
        if i == len(golden_matrix) - 1:
            new_filter_lines.append(f"\t\t({formatted_values})")
        else:
            new_filter_lines.append(f"\t\t({formatted_values}),")

    new_filter_definition = (
        f"constant {filter_name}: FILTER_TYPE:= (\n" + "\n".join(new_filter_lines) + "\n\t);"
    )
    return content.replace(match.group(0), new_filter_definition)


def replace_bias_with_golden_value(content: str, bias_name: str, golden_value: str) -> str:
    """Reemplaza completamente un valor BIAS con el valor golden original."""
    pattern = rf'constant {bias_name}: signed \(\d+ downto 0\) := "([^"]+)";'
    match = re.search(pattern, content)

    if not match:
        print(f"⚠️ No se encontró el bias {bias_name}")
        return content

    print(f"✅ Encontrado bias {bias_name}")
    print(f"🔄 Valor actual: {match.group(1)}")
    print(f"🔄 Valor golden: {golden_value}")

    bit_width = len(match.group(1))
    new_bias_definition = f'constant {bias_name}: signed ({bit_width - 1} downto 0) := "{golden_value}";'
    return content.replace(match.group(0), new_bias_definition)


def inject_golden_values_to_vhdl(content: str) -> str:
    """Inyecta los valores golden (originales) en el contenido VHDL."""
    from vhdl_hardware.golden_values import get_all_golden_values

    try:
        print("🔄 Iniciando inyección de valores golden...")
        golden_values = get_all_golden_values()

        for i in range(1, 7):
            filter_name = f"FMAP_{i}"
            if filter_name in golden_values["fmap"]:
                content = replace_filter_with_golden_values(content, filter_name, golden_values["fmap"][filter_name])
                print(f"✅ Valor golden inyectado para {filter_name}")

        for i in range(1, 7):
            bias_name = f"BIAS_VAL_{i}"
            if bias_name in golden_values["bias"]:
                content = replace_bias_with_golden_value(content, bias_name, golden_values["bias"][bias_name])
                print(f"✅ Valor golden inyectado para {bias_name}")

        print("✅ Inyección de valores golden completada")
        return content

    except Exception as e:
        print(f"❌ ERROR en inyección de valores golden: {str(e)}")
        raise Exception(f"Error inyectando valores golden: {str(e)}")
