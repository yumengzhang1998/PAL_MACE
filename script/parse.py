import re


def clean_bi_log(input_file, output_file):
    # Pattern 1: Oracle label output
    oracle_pattern = re.compile(r"^rank \d+ done in \d+\.\d+ seconds, \d+ labels with \(\d+,\) shape generated$")
    # Pattern 2: Rank running line
    running_pattern = re.compile(r"^rank \d+ running$")
    # Pattern 3: Any line with 'epoch' or 'Epoch'
    epoch_pattern = re.compile(r".*epoch.*", re.IGNORECASE)
    md_pattern = re.compile(r".*MD.*", re.IGNORECASE)
    info_pattern = re.compile(r".*INFO.*", re.IGNORECASE)

    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    filtered_lines = [
        line for line in lines
        if not (
            oracle_pattern.match(line.strip()) or
            running_pattern.match(line.strip()) or
            epoch_pattern.search(line) or
            md_pattern.search(line) or
            info_pattern.search(line)
        )
    ]

    with open(output_file, 'w') as outfile:
        outfile.writelines(filtered_lines)

if __name__ == '__main__':
    clean_bi_log('bi4-6.txt', 'bi4-6_cleaned.txt')
