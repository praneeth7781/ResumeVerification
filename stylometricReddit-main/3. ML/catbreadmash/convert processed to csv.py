import csv


def preprocess_text(text):
    return text.strip()


def makeCSV(input_file, output_file, reddit_user):
    with open(input_file, "r", encoding="utf-8") as file:
        text_data = file.read()

    # split the comments by new lines, very crude method of doing so, will refine further
    comments = text_data.split("\n")

    data_list = []
    for comment in comments:
        processed_comment = preprocess_text(comment)
        if processed_comment:  # Skip empty comments
            data_list.append({"text": processed_comment, "author": reddit_user})

    with open(output_file, "w", newline="", encoding="utf-8") as file:
        fieldnames = ["text", "author"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(data_list)


if __name__ == "__main__":
    input_file = "proc.txt"
    output_file = "../0. combined/catbreadmash.csv"  # change the name of the csv file to whichever author you're testing
    reddit_user = "catbreadmash"  # have to manually change username for each author

    makeCSV(input_file, output_file, reddit_user)
