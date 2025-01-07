from langchain_core.messages import ToolMessage


# print_colored:
# Función que te permite imprimir texto coloreado en la terminal.
#
# Args:
# - text (str): Texto que deseas imprimir.
# - color_code (int): Código del color que quieres aplicar.
#   A continuación, se detallan los códigos de colores disponibles:
#
#   **Colores estándar:**
#   - 30: Negro
#   - 31: Rojo
#   - 32: Verde
#   - 33: Amarillo
#   - 34: Azul
#   - 35: Magenta
#   - 36: Cian
#   - 37: Blanco
#
#   **Colores brillantes (intensos):**
#   - 90: Negro brillante
#   - 91: Rojo brillante
#   - 92: Verde brillante
#   - 93: Amarillo brillante
#   - 94: Azul brillante
#   - 95: Magenta brillante
#   - 96: Cian brillante
#   - 97: Blanco brillante
#
#   **Estilos adicionales:**
#   - 1: Negrita (agregable como prefijo al color, e.g., '1;31' para rojo en negrita)
#   - 4: Subrayado (usado junto con colores, e.g., '4;32' para verde subrayado)
#   - 0: Resetear estilos (se usa automáticamente al final de la impresión)
#
# Ejemplo de uso:
# print_colored("Hola Mundo", 31)  # Imprime "Hola Mundo" en rojo.
# print_colored("¡Éxito!", 92)     # Imprime "¡Éxito!" en verde brillante.
def print_colored(text, color_code):
    """
    Imprime texto en colores específicos en la terminal.

    :param text: El texto que deseas imprimir (str).
    :param color_code: El código del color ANSI que define el color del texto (int).
                       Códigos de colores permitidos:
                       - 30: Negro
                       - 31: Rojo
                       - 32: Verde
                       - 33: Amarillo
                       - 34: Azul
                       - 35: Magenta
                       - 36: Cian
                       - 37: Blanco
                       - 90: Negro brillante
                       - 91: Rojo brillante
                       - 92: Verde brillante
                       - 93: Amarillo brillante
                       - 94: Azul brillante
                       - 95: Magenta brillante
                       - 96: Cian brillante
                       - 97: Blanco brillante

    :return: None
    """
    print(f"\033[{color_code}m{text}\033[0m")

