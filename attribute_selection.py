import numpy as np

# TODO: document

def forward_attribute_selector(y, x, evaluator, 
        attribute_subset_size, verbose=False):
    num_instances, num_attributes = x.shape
    avaliable_attributes = list(range(num_attributes))
    choosen_attributes = []
    x_subset = np.empty((num_instances, 0), float)
    while len(avaliable_attributes) > 0 and \
            len(choosen_attributes) < attribute_subset_size:
        current_best = {
            'index' : None,
            'score' : None
        }
        for attr_index in avaliable_attributes:
            x_subset = np.append(x_subset, x[:, [attr_index]], axis=1)
            score = evaluator(y, x_subset)
            if current_best['score'] is None or \
                    score > current_best['score']:
                current_best = {
                    'index' : attr_index,
                    'score' : score
                }
            x_subset = np.delete(x_subset, -1, axis=1)

        choosen_attributes.append(current_best['index'])
        avaliable_attributes.remove(current_best['index'])
        x_subset = np.append(x_subset, 
                x[:, [current_best['index']]],
                axis = 1)

        if verbose:
            best_score = current_best['score']
            best_attributes = ', '.join(map(str, choosen_attributes))
            print(f' --- FORWARD ATTRIBUTE SELECTION: ' + 
                f'Best attribute indexes for size {len(choosen_attributes)}: ' +
                f'{best_attributes} (score {best_score:.6})')
    
    choosen_attributes.sort()
    return choosen_attributes