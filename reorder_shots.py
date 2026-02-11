import csv
import sys

def reorder_csv_by_shot(input_file, output_file):
    """
    Read CSV file, reorder by shot number chronologically, 
    and add a blank row between each unique shot.
    """
    # Read all data
    rows = []
    current_shot = None
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip completely empty rows
            if not any(row.values()):
                continue
                
            shot = row['Shot'].strip()
            if shot:  # If shot is not empty
                current_shot = shot
                rows.append(row)
            else:
                # This handles rows with empty Shot but other data
                rows.append(row)
    
    # Sort by shot number
    def get_shot_number(row):
        shot = row['Shot'].strip()
        if shot:
            try:
                return int(shot)
            except ValueError:
                return float('inf')  # Put non-numeric shots at the end
        return float('inf')
    
    rows.sort(key=get_shot_number)
    
    # Write output with blank rows between unique shots
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['Shot', 't_start', 't_end', 'Label']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        previous_shot = None
        for row in rows:
            current_shot = row['Shot'].strip()
            
            # Add blank row when shot changes
            if current_shot and previous_shot is not None and current_shot != previous_shot:
                writer.writerow({})
            
            writer.writerow(row)
            previous_shot = current_shot
    
    print(f"Successfully created {output_file} with shots in chronological order")

if __name__ == '__main__':
    input_file = 'Random/Copy of DIII-D ELM Control Classification Label Database - Sheet1 (1).csv'
    output_file = 'Random/Copy of DIII-D ELM Control Classification Label Database - Sheet1 (1) - reordered.csv'
    
    try:
        reorder_csv_by_shot(input_file, output_file)
        print(f"Output saved to: {output_file}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

