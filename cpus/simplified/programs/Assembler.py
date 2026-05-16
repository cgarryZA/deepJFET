import os

# Assembler
def first_pass(lines, opcodes):
    addressCounter = 0
    for linenumber, line in enumerate(lines, 1):
        words = line.strip().split()

        opcode = words[0]
        
        # If opcode doesn't exist in the predefined dictionary, treat it as a label
        if not opcodes.get_opcode(opcode):
            label = opcode
            line_binary = bin(addressCounter)[2:].zfill(12)
            formatted_line = f"{line_binary[0:4]} {line_binary[4:8]} {line_binary[8:]}"
            opcodes.add_opcode(label, formatted_line)
            #print(f"Label: {opcode} Address: {addressCounter}")
            opcode = words[1]
        
        if opcode == 'JCN':
            addressCounter += 2
        elif opcode == 'FIM':
            addressCounter += 2
        elif opcode == 'JUN':
            addressCounter += 2
        elif opcode == 'JMS':
            addressCounter += 2
        elif opcode == 'ISZ':
            addressCounter += 2
        else:
            addressCounter += 1
  
def assembler():
    opcodes = OpcodeMapping()
    Instructs = DefaultOpcodes()

    with open('Assembled.asm', 'r') as fileID:
        lines = fileID.readlines()

    first_pass(lines, opcodes)

    with open('Machine.bin', 'w') as outputFileID:
        for lineNumber, line in enumerate(lines, 1):
            words = line.strip().split()

            if len(words) > 1:
                if Instructs.get_opcode(words[1]):
                    words.pop(0)

            opcode = words[0]

            #print(f"Processing line with opcode: {opcode}")

            if opcode == 'NOP':
                check_length(lineNumber, words, 1)
                outputFileID.write(f'{opcodes.get_opcode(opcode)}\n')
            elif opcode == 'JCN':
                check_length(lineNumber, words, 3)  # Change to 3 as minimum number of words
                cccc = convert_value(words[1])
                if words[2].isdigit():  # It's an integer
                    aaaa = convert_value(words[2]) + ' ' + convert_value(words[3])
                else:  # It's a label
                    # Extract the label value
                    label_value = opcodes.get_opcode(words[2])
                    if not label_value:
                        print("Undefined Label Line: "f'{lineNumber}')
                        break
                    # Only take the bottom 8 bits
                    aaaa = label_value.split()[1] + ' ' + label_value.split()[2]
                    outputFileID.write(f'{opcodes.get_opcode(opcode)} {cccc} {aaaa}\n')

            elif opcode == 'FIM':
                check_length(lineNumber, words, 4)
                outputFileID.write(f'{opcodes.get_opcode(opcode)} {convert_value(words[1], 3)}0 {convert_value(words[2])} {convert_value(words[3])}\n')
            elif opcode == 'SRC':
                check_length(lineNumber, words, 2)
                outputFileID.write(f'{opcodes.get_opcode(opcode)} {convert_value(words[1], 3)}1\n')
            elif opcode == 'FIN':
                check_length(lineNumber, words, 2)
                outputFileID.write(f'{opcodes.get_opcode(opcode)} {convert_value(words[1], 3)}0\n')
            elif opcode == 'JIN':
                check_length(lineNumber, words, 2)
                outputFileID.write(f'{opcodes.get_opcode(opcode)} {convert_value(words[1], 3)}1\n')

            elif opcode == 'JUN':

                if len(words) == 2:
                    label = words[1]
                    label_value = opcodes.get_opcode(label)
                    outputFileID.write(f'{opcodes.get_opcode(opcode)} {label_value}\n')
                else:
                    check_length(lineNumber, words, 4)
                    outputFileID.write(f'{opcodes.get_opcode(opcode)} {convert_value(words[1])} {convert_value(words[2])} {convert_value(words[3])}\n')
            
            elif opcode == 'JMS':

                if len(words) == 2:
                    #print(f"Processing JMS with label: {words[1]}")
                    label = words[1]
                    label_value = opcodes.get_opcode(label)
                    outputFileID.write(f'{opcodes.get_opcode(opcode)} {label_value}\n')
                else:
                    #print(f"Processing JMS with multiple operands: {words[1:]}")
                    check_length(lineNumber, words, 4)
                    outputFileID.write(f'{opcodes.get_opcode(opcode)} {convert_value(words[1])} {convert_value(words[2])} {convert_value(words[3])}\n')
            
            elif opcode == 'INC':
                check_length(lineNumber, words, 2)
                outputFileID.write(f'{opcodes.get_opcode(opcode)} {convert_value(words[1])}\n')
            elif opcode == 'ISZ':
                check_length(lineNumber, words, 3)  # Change to 3 as minimum number of words
                rrrr = convert_value(words[1])
                if words[2].isdigit():  # It's an integer
                    aaaa = convert_value(words[2]) + ' ' + convert_value(words[3])
                else:  # It's a label
                    # Extract the label value
                    label_value = opcodes.get_opcode(words[2])
                    if not label_value:
                        print("Undefined Label Line: "f'{lineNumber}')
                        break
                    # Only take the bottom 8 bits
                    aaaa = label_value.split()[1] + ' ' + label_value.split()[2]
                    outputFileID.write(f'{opcodes.get_opcode(opcode)} {rrrr} {aaaa}\n')
            elif opcode in ['ADD', 'SUB', 'LD', 'XCH']:
                check_length(lineNumber, words, 2)
                outputFileID.write(f'{opcodes.get_opcode(opcode)} {convert_value(words[1])}\n')
            elif opcode in ['BBL', 'LDM']:
                check_length(lineNumber, words, 2)
                outputFileID.write(f'{opcodes.get_opcode(opcode)} {convert_value(words[1])}\n')
            elif opcode in ['CLB', 'CLC', 'IAC', 'CMC', 'CMA', 'RAL', 'RAR', 'TCC', 'DAC', 'TCS', 'STC', 'DAA', 'KBP', 'DCL']:
                check_length(lineNumber, words, 1)
                outputFileID.write(f'{opcodes.get_opcode(opcode)}\n')
            elif opcode in ['WRM', 'WMP', 'WRR', 'WPM', 'WR0', 'WR1', 'WR2', 'WR3', 'SBM', 'RDM', 'RDR', 'ADM', 'RD0', 'RD1', 'RD2', 'RD3']:
                check_length(lineNumber, words, 1)  
                outputFileID.write(f'{opcodes.get_opcode(opcode)}\n')

def binary_to_hex(binary_input):
    # Removing all spaces and newline characters from the input
    binary_input = binary_input.replace(" ", "").replace("\n", "")

    hex_output = ""
    char_count = 0  # Counter for characters in the current line

    # Processing 4 bits at a time
    for i in range(0, len(binary_input), 4):
        four_bits = binary_input[i:i+4]

        # Converting 4 bits to hexadecimal
        hex_digit = format(int(four_bits, 2), 'X')

        # Adding the hex digit to the output
        hex_output += hex_digit
        char_count += len(hex_digit)

        # Adding a space after every hex digit pair and a newline after every 16 characters
        if i + 4 < len(binary_input):
            if char_count % 2 == 0:
                hex_output += " "
            if char_count % 16 == 0:
                hex_output += "\n"

    return hex_output.strip()

def convert_value(inputWord, numBits=4):
    if inputWord.startswith('b'):
        value = inputWord[1:]
    elif inputWord.startswith('h'):
        value = bin(int(inputWord[1:], 16))[2:]
    else:
        value = bin(int(inputWord))[2:]

    if len(value) < numBits:
        value = '0' * (numBits - len(value)) + value
    elif len(value) > numBits:
        raise ValueError(f'Error on line : Invalid value length for {inputWord}')

    if any(char != '0' and char != '1' for char in value):
        raise ValueError(f'Error on line : Invalid characters in {inputWord}')

    return value

def check_length(lineNumber, words, expectedLength):
    if len(words) != expectedLength:
        raise ValueError(f'Error on line {lineNumber}: Expected {expectedLength} words, found {len(words)}')

class OpcodeMapping:
    def __init__(self):
        self.mapping = {
            'NOP': '0000 0000',
            'JCN': '0001',
            'FIM': '0010',
            'SRC': '0010',
            'FIN': '0011',
            'JIN': '0011',
            'JUN': '0100',
            'JMS': '0101',
            'INC': '0110',
            'ISZ': '0111',
            'ADD': '1000',
            'SUB': '1001',
            'LD' : '1010',
            'XCH': '1011',
            'BBL': '1100',
            'LDM': '1101',

            'WRM': '1110 0000',
            'WMP': '1110 0001',
            'WRR': '1110 0010',
            'WPM': '1110 0011',
            'WR0': '1110 0100',
            'WR1': '1110 0101',
            'WR2': '1110 0110',
            'WR3': '1110 0111',
            'SBM': '1110 1000',
            'RDM': '1110 1001',
            'RDR': '1110 1010',
            'ADM': '1110 1011',
            'RD0': '1110 1100',
            'RD1': '1110 1101',
            'RD2': '1110 1110',
            'RD3': '1110 1111',

            'CLB': '1111 0000',
            'CLC': '1111 0001',
            'IAC': '1111 0010',
            'CMC': '1111 0011',
            'CMA': '1111 0100',
            'RAL': '1111 0101',
            'RAR': '1111 0110',
            'TCC': '1111 0111',
            'DAC': '1111 1000',
            'TCS': '1111 1001',
            'STC': '1111 1010',
            'DAA': '1111 1011',
            'KBP': '1111 1100',
            'DCL': '1111 1101'
        }

    def get_opcode(self, key):
        return self.mapping.get(key)
    
    def add_opcode(self, key, value):
        if key not in self.mapping:
            self.mapping[key] = value
        else:
            print(f"Opcode {key} already exists")

class DefaultOpcodes:
    def __init__(self):
        self.mapping = {
            'NOP': '0000 0000 0000 0000',
            'JCN': '0001',
            'FIM': '0010',
            'SRC': '0010',
            'FIN': '0011',
            'JIN': '0011',
            'JUN': '0100',
            'JMS': '0101',
            'INC': '0110',
            'ISZ': '0111',
            'ADD': '1000',
            'SUB': '1001',
            'LD': '1010',
            'XCH': '1011',
            'BBL': '1100',
            'LDM': '1101',

            'WRM': '1110 0000',
            'WMP': '1110 0001',
            'WRR': '1110 0010',
            'WPM': '1110 0011',
            'WR0': '1110 0100',
            'WR1': '1110 0101',
            'WR2': '1110 0110',
            'WR3': '1110 0111',
            'SBM': '1110 1000',
            'RDM': '1110 1001',
            'RDR': '1110 1010',
            'ADM': '1110 1011',
            'RD0': '1110 1100',
            'RD1': '1110 1101',
            'RD2': '1110 1110',
            'RD3': '1110 1111',

            'CLB': '1111 0000',
            'CLC': '1111 0001',
            'IAC': '1111 0010',
            'CMC': '1111 0011',
            'CMA': '1111 0100',
            'RAL': '1111 0101',
            'RAR': '1111 0110',
            'TCC': '1111 0111',
            'DAC': '1111 1000',
            'TCS': '1111 1001',
            'STC': '1111 1010',
            'DAA': '1111 1011',
            'KBP': '1111 1100',
            'DCL': '1111 1101'
        }

    def get_opcode(self, key):
        return self.mapping.get(key)

# Writing to ROM
def generate_pwl_files(data, output_dir="PROGRAM"):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process input data into a list, each item will be a line in a file
    pwl_lines = []

    for line in data.strip().split('\n'):
        words = line.split()
        for word in words:
            if len(word) != 4:
                raise ValueError(f"Word '{word}' is not 4 bits long.")
            for bit in word:
                pwl_lines.append("0u -0.8\n" if bit == '1' else "0u -3.6\n")

    # Write each line to a separate file, named sequentially
    for index, content in enumerate(pwl_lines):
        filename = f"{index}.txt"
        file_path = os.path.join(output_dir, filename)

        # Write to the PWL file
        with open(file_path, 'w') as file:
            file.write(content)

    # Write additional files up to 2047.txt with "0u -3.6"
    for index in range(len(pwl_lines), 16385):
        filename = f"{index}.txt"
        file_path = os.path.join(output_dir, filename)
        with open(file_path, 'w') as file:
            file.write("0u -3.6\n")

def read_binary_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data

def save_to_file(data, filename):
    with open(filename, 'w') as file:
        file.write(data)

if __name__ == "__main__":

    assembler()
    
    input_data = read_binary_file('Machine.bin')

    # Convert the binary input to hexadecimal, ignoring whitespace
    hex_output = binary_to_hex(input_data.strip())
    save_to_file(hex_output, 'Machine.hex')

    generate_pwl_files(input_data)