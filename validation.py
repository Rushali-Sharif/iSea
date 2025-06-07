import os
import re

def replace_numbers(folder_path):
    # Define the replacement dictionary
    replacements = {'0': '0','1': '1','2': '2','3': '3','4': '4','5': '5','6': '6','7': '7','8': '8','9': '9','10': '10','11': '11','12': '13','13': '14','14': '15','15': '16','16': '17','17': '18','18': '19','19': '20','20': '21','21': '23','22': '24','23': '25','24': '26','25': '27','26': '28','27': '29','28': '30','29': '31','30': '32','31': '33','32': '34'}
    # Compile the regex to match 0, 1, or 2
    pattern = re.compile(r'\b(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33)\b')
    # Function to replace match with corresponding value from the dictionary
    def replace_match(match):
        return replacements[match.group(0)]
    
    # Iterate through all files in the given folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Process only text files
        if os.path.isfile(file_path) and filename.endswith('.txt'):
            with open(file_path, 'r') as file:
                content = file.read()
            
            # Replace the numbers in the file
            new_content = pattern.sub(replace_match, content)
            
            # Write the modified content back to the file
            with open(file_path, 'w') as file:
                file.write(new_content)
                
            print(f'Updated {filename}')

# Usage
folder_path = 'D:/objectDetection/objectDetection/Object Detection.v4i.yolov5pytorch/test/labels/'  # Specify the folder containing text files
replace_numbers(folder_path)
