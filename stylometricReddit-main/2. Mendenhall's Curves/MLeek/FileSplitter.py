import os

def split_text_file(input_file, output_directory):
    with open(input_file, 'r') as file:
        content = file.read()

    words = content.split()
    total_words = len(words)

    num_files = total_words // 2000
    if total_words % 2000 != 0:
        num_files += 1

    for i in range(num_files):
        start = i * 2000
        end = start + 2000
        file_content = ' '.join(words[start:end])

        output_file = os.path.join(output_directory, f'output_{i+1}.txt')
        with open(output_file, 'w') as file:
            file.write(file_content)

        print(f'Created {output_file} with {len(file_content.split())} words.')


input_file_path = 'proc.txt'
output_directory_path = './splitup/'

split_text_file(input_file_path, output_directory_path)