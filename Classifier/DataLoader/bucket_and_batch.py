
import numpy as np
import random
import re


def bucket_and_batch(vectors, classes, batch_size, classes_num):

    print("Sample size: ", vectors.shape[0])

    i = 0
    batches_vectors = []
    batches_classes = []
    count = 0

    while i < len(vectors):

        if i+batch_size > vectors.shape[0]:
            batch_size = vectors.shape[0]-i

        batch_vectors = vectors[i:i+batch_size]
        batch_classes = []

        for j in range(i, i + batch_size):
            class_ = classes[j].tolist()
            # print(class_)
            new_class = [1 if x in class_ else 0 for x in range(classes_num)]
            # print(new_class)
            batch_classes.append(new_class)

        batch_classes = np.asarray(batch_classes, dtype=int)

        batches_vectors.append(batch_vectors)
        batches_classes.append(batch_classes)

        i += batch_size

    return batches_vectors, batches_classes
