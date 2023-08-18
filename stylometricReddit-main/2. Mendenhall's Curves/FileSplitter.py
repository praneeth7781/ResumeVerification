import os

def split_text_file(input_file, output_directory, words_per_file):
    with open(input_file, 'r') as file:
        content = file.read()

    words = content.split()
    total_words = len(words)

    num_files = total_words // words_per_file
    if total_words % words_per_file != 0:
        num_files += 1

    for i in range(num_files):
        start = i * words_per_file
        end = start + words_per_file
        file_content = ' '.join(words[start:end])

        output_file = os.path.join(output_directory, f'output_{i+1}.txt')
        with open(output_file, 'w') as file:
            file.write(file_content)

        print(f'Created {output_file} with {len(file_content.split())} words.')


# Usage example
input_file_path = '/proc.txt'
output_directory_path = '/splitup'
words_per_file = 2000

split_text_file(input_file_path, output_directory_path, words_per_file)