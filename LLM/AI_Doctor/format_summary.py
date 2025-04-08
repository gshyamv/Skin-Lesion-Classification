# Function to replace '\n' with '<br>'
def replace_newline_with_br(input_string):
    return input_string.replace('\n', '\n\n')

def replace_t_with_tab(input_string):
    return input_string.replace('\t', '    ')