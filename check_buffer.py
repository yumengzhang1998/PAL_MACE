import json
import numpy as np

def check_entry(entry, atom_number=11):
    """
    Returns a dict with validation status of each segment
    """
    i = 0
    result = {}

    try:
        # Coordinates
        coords = entry[i:i+atom_number*3]
        result['coordinates'] = len(coords) == atom_number * 3
        i += atom_number * 3

        # Atomic numbers
        atomic_numbers = entry[i:i+atom_number]
        result['atomic_numbers'] = len(atomic_numbers) == atom_number and all(int(n) == 83 for n in atomic_numbers)
        i += atom_number

        # True energy
        true_energy = entry[i]
        result['true_energy'] = isinstance(true_energy, (float, int))
        i += 1

        # True forces
        true_forces = entry[i:i+atom_number*3]
        result['true_forces'] = len(true_forces) == atom_number * 3
        i += atom_number * 3

        # Charge
        charge = entry[i]
        result['charge'] = isinstance(charge, (int, float))  # you can enforce int if needed
        i += 1

        # Predicted forces
        pred_forces = entry[i:i+atom_number*3]
        result['pred_forces'] = len(pred_forces) == atom_number * 3
        i += atom_number * 3

        # Predicted energy
        pred_energy = entry[i]
        result['pred_energy'] = isinstance(pred_energy, (float, int))
        i += 1

        # Patience
        patience = entry[i:i+2]
        result['patience'] = len(patience) == 2 and all(isinstance(p, (int, float)) for p in patience)
        i += 2

        # Velocities
        velocities = entry[i:i+atom_number*3]
        result['velocities'] = len(velocities) == atom_number * 3
        i += atom_number * 3

        # Final check
        result['total_length'] = len(entry) == i
    except Exception as e:
        result['error'] = str(e)

    return result


# Load and check all entries
with open("to_orcl_buffer.json", "r") as f:
    data = json.load(f)

bad_entries = []
for idx, entry in enumerate(data):
    check = check_entry(entry)
    if not all(check.values()):
        bad_entries.append((idx, check))

# Report
print(f"Total entries checked: {len(data)}")
print(f"Malformed entries found: {len(bad_entries)}")
for idx, problems in bad_entries:  # show only first 5 for brevity
    print(f"\nIndex {idx} has issues:")
    print(data[idx])
    for k, v in problems.items():
        if not v:
            print(f"  ❌ {k} failed")

