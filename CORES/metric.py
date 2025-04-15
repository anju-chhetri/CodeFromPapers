import numpy as np

def batch_jaccard_similarity(filter1, filter2):
    """
    Calculate Jaccard similarity between filters of top 2 classes for multiple samples
    
    Args:
        filters_array: numpy array of shape (n_samples, n_filters) containing filter indices
        Here shape would be (600, 10)
    
    Returns:
        numpy array of shape (n_samples,) containing Jaccard similarities
    """
    # Assuming first half of each row is for class 1 and second half for class 2
    # If that's not the case, modify the split point accordingly
    # split_point = filters_array.shape[1] // 2
    
    # class1_filters = filters_array[:, :split_point]
    # class2_filters = filters_array[:, split_point:]
    
    # # Vectorized intersection and union
    # similarities = np.zeros(len(filters_array))
    
    # for i in range(len(filters_array)):
    #     set1 = set(class1_filters[i])
    #     set2 = set(class2_filters[i])
        
    #     intersection = len(set1.intersection(set2))
    #     union = len(set1.union(set2))
        
    #     similarities[i] = intersection / union if union != 0 else 0
        # Find common elements across dimension 1
    
    common_elements = [len(np.intersect1d(filter1[i, :], filter2[i, :])) for i in range(filter1.shape[0])]
    # Convert the list of arrays to a numpy array for better handling
    common_elements = np.array(common_elements, dtype=object)

    # common_elements = np.asarray([elem if elem.size > 0 else np.array([0]) for elem in common_elements])
    
    return common_elements/filter1.shape[1]