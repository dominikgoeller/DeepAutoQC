import os
import uuid
import csv

# TODO: OOP: create class with path as constructor parameter which was given as input by user

source_path = '/Users/dominik/University/Bachelor Thesis/test test data/reports' # Source Path currently needs to have the example reports tree structure!

def rename_sub_to_id():
    """
    Rename all folders and .svg files of a 'reports' batch which match a specific pattern with an unique ID to pseudonymize subjects
    Save IDs and original filename to a CSV file.
    """
    # ext = ('.svg', '.png', '.jpeg') maybe give ext as argument for user to decide assuming there will be different files than .svg ELSE throw away

    for dir in os.listdir(source_path):
        # Looping through all "sub-id" folders while ignoring .DS_Store file
        # Here we need to: 
        # - Generate new ID
        # - Save the original sub-id String of folder
        # - push original sub-id String and new ID into CSV/whatever file
        # Check if folder is directory in case 
        if (dir.startswith('.DS_Store') or dir.endswith('.csv')):
            continue
        if not os.path.isdir(os.path.join(source_path,dir)):
            continue

        pseudo_ID = uuid.uuid4().hex # uuid4 safe enough (?) collision check necessary at which number of subjects!?
        original_sub_id = dir
        with open(os.path.join(source_path, 'ID_pairs'+'.csv'), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([original_sub_id] + [pseudo_ID])
        
        if not os.path.isdir(os.path.join(source_path,dir)):
            continue

        os.rename(os.path.join(source_path,dir), os.path.join(source_path, pseudo_ID))
        new_folder_path = os.path.join(source_path,pseudo_ID)
        for root, dirs, files in os.walk(new_folder_path):
            for file in files:
                if not file.endswith('.svg'):
                    continue

                os.replace(os.path.join(root,file), 
                    os.path.join(root, 
                    file.replace(original_sub_id, pseudo_ID)))
                            
if __name__ == '__main__':
    rename_sub_to_id()