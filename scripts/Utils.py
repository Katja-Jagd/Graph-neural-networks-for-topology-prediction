import torch
import torch.nn as nn
import math
from typing import List, Union
from Dataset import *
import numpy as np
from torch import tensor
from torchmetrics.classification import MulticlassAUROC

def cv_fold_split(data_dict, cv_values):
    """
    Filters keys from the dictionary based on specific cv values.

    Parameters:
    - data_dict (dict): The dictionary to filter.
    - cv_values (list of int): The list of cv values to filter by.

    Returns:
    - list: A list of keys that have any of the specified cv values.
    """
    # Filter the keys based on the 'cv' value
    proteins = [key for key, values in data_dict.items() if values.get('cv') in cv_values]

    return proteins

def labels_to_tensors(labels, device):
    """
    Input: list of labels as strings and translation dictionary 
    {'I': 0, 'O':1, 'P': 2, 'S': 3, 'M':4, 'B': 5}, specify train/test mode
    Output: A tuple of tensors for each string with the numeric encoding 
    """
    translation_dict = {'I': 0, 'O':1, 'P': 2, 'S': 3, 'M':4, 'B': 5} 

    tensor_list = []
    for label in labels:
        # Map each character in the label to its corresponding value 
        mapped_values = [translation_dict[char] for char in label]
        # Convert the list of mapped values into a PyTorch tensor
        tensor = torch.tensor(mapped_values, dtype=torch.long)
        tensor_list.append(tensor)
    
    # Convert the list of tensors into a tuple
    tensor_tuple = tuple(tensor.to(device) for tensor in tensor_list)
    
    return tensor_tuple

def train_one_epoch(model, optimizer, criterion, trainset, device, decoder_type, result_type):
    """
    Input: Trainset loaded in batches
    Output: train loss and train accuracy per epoch
    """
    # Initialization 
    model.train()
    loss_per_batch = 0
    all_predictions = []
    all_labels = []
    num_batches = len(trainset)
    c = 0
    # Loop over all batches 
    for batch in trainset:
        c += 1
        print(f' Training Batch: {c}')
         
        batch.to(device)
        batch.num_relation = 2 # temporary 
            
        # Forward pass
        output, protein_lengths = model(batch)
        batch_size = len(output)

        # Creating tuples for each protein, for output and target labels   
        # CHANGE LATER SO BOTH DECODERS HAVE SAME FORMAT 
        if decoder_type == 'linear':
            output_per_protein = torch.split(output, protein_lengths)
        elif decoder_type == 'lstm':
            output_per_protein = output
        targets = labels_to_tensors(batch.label, device)

        loss_per_protein = 0
        # Loop over all proteins in one batch
        for output_, target_ in zip(output_per_protein, targets):
            #target_.to(device)
            loss_per_protein += criterion(output_, target_)
            
            # Getting predition for one protein
            pred_per_protein = torch.argmax(torch.nn.Softmax(dim=1)(output_), dim=1)

            # Save prediction and label for accuracy evaluation 
            all_predictions.append(pred_per_protein)
            all_labels.append(target_)
        
        # Claculate average loss for current batch 
        loss = loss_per_protein / batch_size
        loss_per_batch += loss

        # Clear old gradients
        optimizer.zero_grad()

        # Compute gradients based on the loss from the current batch (backpropagation)
        loss.backward()

        # Take one optimizer step using the gradients computed in the previous step.
        optimizer.step()

    # Calculate average loss for all batches 
    loss_per_epoch = loss_per_batch / num_batches
    

    """
    # USING THEIR METRICS
    # Calculate accuracies for all batches
    overall_performance = bio_acc(all_predictions, all_labels)
    top_acc = overall_performance['top_acc']
    sptm_top_acc = overall_performance["sptm_top_acc"]
    tm_top_acc = overall_performance['tm_top_acc']
    sp_top_acc = overall_performance["sp_top_acc"]
    glob_top_acc = overall_performance["glob_top_acc"]
    beta_top_acc = overall_performance["beta_top_acc"]
    
    # Return loss_per_epoch and topology acc_per_epoch 
    return loss_per_epoch, top_acc, sptm_top_acc, tm_top_acc, sp_top_acc, glob_top_acc, beta_top_acc 
    
    # USING OWN METRICS
    # Calculate accuracies for all batches
    
    overall_performance = bio_top_residue_acc(all_predictions, all_labels, device, result_type)
    
    average_overlap_acc = overall_performance['average_overlap_acc'] 
    tm_overlap_acc = overall_performance['tm_overlap_acc']
    sptm_overlap_acc = overall_performance["sptm_overlap_acc"]
    sp_overlap_acc = overall_performance["sp_overlap_acc"]
    glob_overlap_acc = overall_performance["glob_overlap_acc"]
    beta_overlap_acc = overall_performance["beta_overlap_acc"]

    average_residue_acc = overall_performance['average_residue_acc'] 
    tm_residue_acc = overall_performance['tm_residue_acc']
    sptm_residue_acc = overall_performance["sptm_residue_acc"]
    sp_residue_acc = overall_performance["sp_residue_acc"]
    glob_residue_acc = overall_performance["glob_residue_acc"]
    beta_residue_acc = overall_performance["beta_residue_acc"]
    """

    if result_type == ['residue']:
        per_residue_acc, confusion_matrix_residue = bio_top_residue_acc(all_predictions, all_labels, device, result_type)
        return loss_per_epoch, per_residue_acc, confusion_matrix_residue
    elif result_type == ['class']: 
        overlap_acc, residue_acc = bio_top_residue_acc(all_predictions, all_labels, device, result_type)
        return loss_per_epoch, overlap_acc, residue_acc
    else:
        overlap_acc, residue_acc, per_residue_acc, confusion_matrix_residue = bio_top_residue_acc(all_predictions, all_labels, device, result_type)
        return loss_per_epoch, overlap_acc, residue_acc, per_residue_acc, confusion_matrix_residue

   # return loss_per_epoch, average_overlap_acc, tm_overlap_acc, sptm_overlap_acc, sp_overlap_acc, glob_overlap_acc, beta_overlap_acc, average_residue_acc, tm_residue_acc, sptm_residue_acc, sp_residue_acc, glob_residue_acc, beta_residue_acc 
    #return loss_per_epoch, overlap_acc, residue_acc

def val_one_epoch(model, criterion, valset, device, decoder_type, result_type):
    """
    Input: Validation set loaded in batches
    Output: Validation loss and accuracy per epoch
    """
    # Initialization 
    loss_per_batch = 0
    all_predictions = []
    all_labels = []
    num_batches = len(valset)

    with torch.no_grad():
        model.eval()
        c = 0 
        # Loop over all batches 
        for batch in valset:
            c += 1
            print(f' Validation Batch: {c}')

            batch.to(device) 
            batch.num_relation = 2 # temporary 
                
            # Forward pass
            output, protein_lengths = model(batch)
            batch_size = len(output)

            # Creating tuples for each protein, for output and target labels   
            # CHANGE LATER SO BOTH DECODERS HAVE SAME FORMAT 
            if decoder_type == 'linear':
                output_per_protein = torch.split(output, protein_lengths)
            elif decoder_type == 'lstm':
                output_per_protein = output
            targets = labels_to_tensors(batch.label, device)
                
            loss_per_protein = 0
            # Loop over all proteins in one batch
            for output_, target_ in zip(output_per_protein, targets):
                loss_per_protein += criterion(output_, target_)
                
                # Getting predition for one protein
                pred_per_protein = torch.argmax(torch.nn.Softmax(dim=1)(output_), dim=1)

                # Save prediction and label for accuracy evaluation 
                all_predictions.append(pred_per_protein)
                all_labels.append(target_)
                
            # Claculate average loss for current batch 
            loss = loss_per_protein / batch_size
            loss_per_batch += loss

    # Calculate average loss for all batches 
    loss_per_epoch = loss_per_batch / num_batches
    
    """
    # USING THEIR METRICS
    # Calculate accuracies for all batches
    overall_performance = bio_acc(all_predictions, all_labels)
    top_acc = overall_performance['top_acc']
    sptm_top_acc = overall_performance["sptm_top_acc"]
    tm_top_acc = overall_performance['tm_top_acc']
    sp_top_acc = overall_performance["sp_top_acc"]
    glob_top_acc = overall_performance["glob_top_acc"]
    beta_top_acc = overall_performance["beta_top_acc"]
    
    # Return loss_per_epoch and topology acc_per_epoch 
    return loss_per_epoch, top_acc, sptm_top_acc, tm_top_acc, sp_top_acc, glob_top_acc, beta_top_acc 
 

    # USING OWN METRICS
    # Calculate accuracies for all batches
    overall_performance = bio_top_residue_acc(all_predictions, all_labels, device, result_type)
  	
    average_overlap_acc = overall_performance['average_overlap_acc'] 
    tm_overlap_acc = overall_performance['tm_overlap_acc']
    sptm_overlap_acc = overall_performance["sptm_overlap_acc"]
    sp_overlap_acc = overall_performance["sp_overlap_acc"]
    glob_overlap_acc = overall_performance["glob_overlap_acc"]
    beta_overlap_acc = overall_performance["beta_overlap_acc"]

    average_residue_acc = overall_performance['average_residue_acc'] 
    tm_residue_acc = overall_performance['tm_residue_acc']
    sptm_residue_acc = overall_performance["sptm_residue_acc"]
    sp_residue_acc = overall_performance["sp_residue_acc"]
    glob_residue_acc = overall_performance["glob_residue_acc"]
    beta_residue_acc = overall_performance["beta_residue_acc"]
    """
    if result_type == ['residue']:
        per_residue_acc, confusion_matrix_residue = bio_top_residue_acc(all_predictions, all_labels, device, result_type)
        return loss_per_epoch, per_residue_acc, confusion_matrix_residue
    elif result_type == ['class']: 
        overlap_acc, residue_acc = bio_top_residue_acc(all_predictions, all_labels, device, result_type)
        return loss_per_epoch, overlap_acc, residue_acc
    else:
        overlap_acc, residue_acc, per_residue_acc, confusion_matrix_residue = bio_top_residue_acc(all_predictions, all_labels, device, result_type)
        return loss_per_epoch, overlap_acc, residue_acc, per_residue_acc, confusion_matrix_residue

def custom_weights_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)

def lazy_initialization(data_dir, data_dict, model, device):
    # Lazy initialization by passing one dummy input through model 
    dummyDset = GearNetDataset(root=data_dir,protein_list=['A4WEW2'],data_dict=data_dict) 
    dummy_loader = DataLoader(dummyDset,batch_size=1,shuffle=False,num_workers=0)
    for data in dummy_loader:
        data.to(device)
        #data.num_relation = 2
        dummy_input = data
        model(dummy_input)

    return 

def label_list_to_topology(labels: Union[List[int], torch.Tensor]) -> List[torch.Tensor]:
    """
    Converts a list of per-position labels to a topology representation.
    This maps every sequence to list of where each new symbol start (the topology), e.g. AAABBBBCCC -> [(0,A),(3, B)(7,C)]

    Parameters
    ----------
    labels : list or torch.Tensor of ints
        List of labels.

    Returns
    -------
    list of torch.Tensor
        List of tensors that represents the topology.
    """

    if isinstance(labels, list):
        labels = torch.LongTensor(labels)

    if isinstance(labels, torch.Tensor):
        zero_tensor = torch.LongTensor([0])
        if labels.is_cuda:
            zero_tensor = zero_tensor.cuda()

        unique, count = torch.unique_consecutive(labels, return_counts=True)
        top_list = [torch.cat((zero_tensor, labels[0:1]))]
        prev_count = 0
        i = 0
        for _ in unique.split(1):
            if i == 0:
                i += 1
                continue
            prev_count += count[i - 1]
            top_list.append(torch.cat((prev_count.view(1), unique[i].view(1))))
            i += 1
        return top_list

def calculate_acc(correct, total):
    total = total.float()
    correct = correct.float()
    if total == 0.0:
        return 1
    return correct / total

def is_topologies_equal(topology_a, topology_b, minimum_seqment_overlap=5):
    """
    Checks whether two topologies are equal.
    E.g. [(0,A),(3, B)(7,C)]  is the same as [(0,A),(4, B)(7,C)]
    But not the same as [(0,A),(3, C)(7,B)]

    Parameters
    ----------
    topology_a : list of torch.Tensor
        First topology. See label_list_to_topology.
    topology_b : list of torch.Tensor
        Second topology. See label_list_to_topology.
    minimum_seqment_overlap : int
        Minimum overlap between two segments to be considered equal.

    Returns
    -------
    bool
        True if topologies are equal, False otherwise.
    """

    if isinstance(topology_a[0], torch.Tensor):
        topology_a = list([a.cpu().numpy() for a in topology_a])
    if isinstance(topology_b[0], torch.Tensor):
        topology_b = list([b.cpu().numpy() for b in topology_b])

    #print("Label topology: ",topology_a)
    #print("Prediction topology: ",topology_b)
    #print("Label topology length: ",len(topology_a))
    #print("Prediction topology length: ",len(topology_b))
    
    if len(topology_a) != len(topology_b): #note: this will return false if the number of topology-switches are incorrect! 
        return False
    for idx, (_position_a, label_a) in enumerate(topology_a):
        if label_a != topology_b[idx][1]:
            if (label_a in (1,2) and topology_b[idx][1] in (1,2)): # assume O == P, ie. we accept if these are interchanged
                continue
            else:
                return False #other topologies: do not accept
        if label_a in (3, 4, 5):
            if label_a == 5:
                # Set minimum segment overlap to 3 for Beta regions
                minimum_seqment_overlap = 3

            if(idx<len(topology_a)-1): #don't want to go beyond the last idx, this raises error
              overlap_segment_start = max(topology_a[idx][0], topology_b[idx][0])
              overlap_segment_end = min(topology_a[idx + 1][0], topology_b[idx + 1][0])
              if overlap_segment_end - overlap_segment_start < minimum_seqment_overlap:
                  return False
            else: #only checking last IDX overlap
               nonOverlapping = max(topology_a[idx][0],topology_b[idx][0])-min(topology_a[idx][0],topology_b[idx][0])
               if nonOverlapping > minimum_seqment_overlap:
                  return False 
    return True

def type_from_labels(label):
    """
    Function that determines the protein type from labels

    Dimension of each label:
    (len_of_longenst_protein_in_batch)

    # Residue class
    0 = inside cell/cytosol (I)
    1 = Outside cell/lumen of ER/Golgi/lysosomes (O)
    2 = periplasm (P)
    3 = signal peptide (S)
    4 = alpha membrane (M)
    5 = beta membrane (B)

    B in the label sequence -> beta
    I only -> globular
    Both S and M -> SP + alpha(TM)
    M -> alpha(TM)
    S -> signal peptide

    # Protein type class
    0 = TM
    1 = SP + TM
    2 = SP
    3 = GLOBULAR
    4 = BETA
    """

    if 5 in label:
        ptype = 4

    elif all(element == 0 for element in label):
        ptype = 3

    elif 3 in label and 4 in label:
        ptype = 1

    elif 3 in label:
       ptype = 2

    elif 4 in label:
        ptype = 0

    elif all(x == 0 or x == -1 for x in label):
        ptype = 3

    else:
        ptype = 5

    return ptype


def bio_acc(output, target):
    """
    Functions that calculates a biological accuracy based on both
    topology and protein type predictions
    """

    confusion_matrix = torch.zeros((6,6),dtype = torch.int64)
    
    for i in range(len(output)):
        
        # Get topology
        output_topology = label_list_to_topology(output[i])
        target_topology = label_list_to_topology(target[i])

        # Get protein type
        output_type = type_from_labels(output[i])
        target_type = type_from_labels(target[i])

        # Check if topologies match
        prediction_topology_match = is_topologies_equal(output_topology, target_topology, 5)

        if target_type == output_type:
            # if we guessed the type right for SP+GLOB or GLOB,
            # count the topology as correct
            if target_type == 2 or target_type == 3 or prediction_topology_match:
                confusion_matrix[target_type][5] += 1
            else:
                confusion_matrix[target_type][output_type] += 1

        else:
            confusion_matrix[target_type][output_type] += 1

 
    # Calculate individual class accuracy for protein type prediction
    tm_type_acc = float(calculate_acc(confusion_matrix[0][0] + confusion_matrix[0][5], confusion_matrix[0].sum()))
    tm_sp_type_acc = float(calculate_acc(confusion_matrix[1][1] + confusion_matrix[1][5], confusion_matrix[1].sum()))
    sp_type_acc = float(calculate_acc(confusion_matrix[2][2] + confusion_matrix[2][5], confusion_matrix[2].sum()))
    glob_type_acc = float(calculate_acc(confusion_matrix[3][3] + confusion_matrix[3][5], confusion_matrix[3].sum()))
    beta_type_acc = float(calculate_acc(confusion_matrix[4][4] + confusion_matrix[4][5], confusion_matrix[4].sum()))

    # Calculate individual class accuracy for protein topology prediction
    tm_top_acc = float(calculate_acc(confusion_matrix[0][5], confusion_matrix[0].sum()))
    sptm_top_acc = float(calculate_acc(confusion_matrix[1][5], confusion_matrix[1].sum()))
    sp_top_acc = float(calculate_acc(confusion_matrix[2][5], confusion_matrix[2].sum()))
    glob_top_acc = float(calculate_acc(confusion_matrix[3][5], confusion_matrix[3].sum()))
    beta_top_acc = float(calculate_acc(confusion_matrix[4][5], confusion_matrix[4].sum()))

    # Calculate average accuracy for protein type prediction
    type_acc= (tm_type_acc + tm_sp_type_acc + sp_type_acc + glob_type_acc + beta_type_acc) / 5

    # Calculate average accuracy for protein topology prediction
    top_acc = (tm_top_acc + sptm_top_acc + sp_top_acc + glob_top_acc + beta_top_acc) / 5

    # Combined accuracy score for topology and type prediction
    total_acc = (type_acc + top_acc) / 2

    result_dict = {"total_acc":total_acc,"top_acc":top_acc,"tm_top_acc":tm_top_acc,"sptm_top_acc":sptm_top_acc,"sp_top_acc":sp_top_acc,"glob_top_acc":glob_top_acc,"beta_top_acc":beta_top_acc,"type_acc":type_acc,
                   "tm_type_acc":tm_type_acc,"tm_sp_type_acc":tm_sp_type_acc,"sp_type_acc":sp_type_acc,"glob_type_acc":glob_type_acc,"beta_type_acc":beta_type_acc}

    return result_dict
"""
def bio_top_residue_acc(output, target, device):

    # Confusion matrix initilization 
    confusion_matrix = torch.zeros((4,5), dtype = torch.int64)
    confusion_matrix_residue = torch.zeros((6,6), dtype = torch.int64)
 
    for i in range(len(output)):
        
        # True class of current protein
        ptype = type_from_labels(target[i])
        
        # Getting topology for output and target 
        output_topology = label_list_to_topology(output[i])
        target_topology = label_list_to_topology(target[i])

        # Checking if topology is equal based on overlap 
        match_tmp = is_topologies_equal(output_topology, target_topology)
        if match_tmp:
            confusion_matrix[0][ptype] += 1
        else:
            confusion_matrix[1][ptype] += 1 
        
        # Checking match in each residue of current protein 
        confusion_matrix[2][ptype] += torch.sum((output[i] == target[i])).item()
        confusion_matrix[3][ptype] += torch.sum((output[i]!= target[i])).item()
        for j in range(len(output[i])):
            row = output[i][j].item() 
            column = target[i][j].item() 
            confusion_matrix_residue[row][column] += 1
 
    # row normalization to get intepretable confusion matrix
    total_per_residue = confusion_matrix_residue.sum(dim=1)
    confusion_matrix_residue = confusion_matrix_residue / total_per_residue
    per_residue_acc = torch.diag(confusion_matrix_residue).tolist()
 
    # Calculating matching overlap accuracy for for each class
    tm_overlap_acc = float(confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[1][0]))
    sptm_overlap_acc = float(confusion_matrix[0][1]/(confusion_matrix[0][1]+confusion_matrix[1][1]))
    sp_overlap_acc = float(confusion_matrix[0][2]/(confusion_matrix[0][2]+confusion_matrix[1][2]))
    glob_overlap_acc = float(confusion_matrix[0][3]/(confusion_matrix[0][3]+confusion_matrix[1][3]))
    beta_overlap_acc = float(confusion_matrix[0][4]/(confusion_matrix[0][4]+confusion_matrix[1][4]))

    average_overlap_acc = (tm_overlap_acc+sptm_overlap_acc+sp_overlap_acc+glob_overlap_acc+beta_overlap_acc)/5

    # Calculating residue accuracy for each class 
    tm_residue_acc = float(confusion_matrix[2][0]/(confusion_matrix[2][0]+confusion_matrix[3][0]))
    sptm_residue_acc = float(confusion_matrix[2][1]/(confusion_matrix[2][1]+confusion_matrix[3][1]))
    sp_residue_acc = float(confusion_matrix[2][2]/(confusion_matrix[2][2]+confusion_matrix[3][2]))
    glob_residue_acc = float(confusion_matrix[2][3]/(confusion_matrix[2][3]+confusion_matrix[3][3]))
    beta_residue_acc = float(confusion_matrix[2][4]/(confusion_matrix[2][4]+confusion_matrix[3][4]))

    average_residue_acc = (tm_residue_acc+sptm_residue_acc+sp_residue_acc+glob_residue_acc+beta_residue_acc)/5
    
    # Collecting results in dict
    results = {'average_overlap_acc': average_overlap_acc,'tm_overlap_acc':tm_overlap_acc,'sptm_overlap_acc':sptm_overlap_acc,'sp_overlap_acc':sp_overlap_acc,'glob_overlap_acc':glob_overlap_acc,
               'beta_overlap_acc':beta_overlap_acc,'average_residue_acc':average_residue_acc,'tm_residue_acc':tm_residue_acc,'sptm_residue_acc':sptm_residue_acc,'sp_residue_acc':sp_residue_acc,
               'glob_residue_acc':glob_residue_acc, 'beta_residue_acc':beta_residue_acc, 'confusion_matrix_residue': confusion_matrix_residue, 'per_residue_acc': per_residue_acc}

    return results
"""

def bio_top_residue_acc(output, target, device, result_type):
    """
    result_type options:
    both: ['class', 'residue']
    one: ['class'] or ['residue']
    """
    print(f'entered the function')
    # matrix initilization 
    if 'class' in result_type:
        print('registered class')
        residue_class_matrix = torch.zeros((4,5), dtype = torch.int64)
    if 'residue' in result_type:
        print('registered residue')
        confusion_matrix_residue = torch.zeros((6,6), dtype = torch.int64)

    for i in range(len(output)):

        if 'class' in result_type:
            # true class of current protein
            ptype = type_from_labels(target[i])

            # getting topology for output and target 
            output_topology = label_list_to_topology(output[i])
            target_topology = label_list_to_topology(target[i])

            # checking if topology is equal based on overlap 
            match_tmp = is_topologies_equal(output_topology, target_topology)
            if match_tmp:
                residue_class_matrix[0][ptype] += 1
            else:
                residue_class_matrix[1][ptype] += 1

            # checking match in each residue of current protein 
            residue_class_matrix[2][ptype] += torch.sum((output[i] == target[i])).item()
            residue_class_matrix[3][ptype] += torch.sum((output[i]!= target[i])).item()

        if 'residue' in result_type:
            for j in range(len(output[i])):
                row = output[i][j].item()
                column = target[i][j].item()
                confusion_matrix_residue[row][column] += 1

    if 'residue' in result_type:
        # row normalization to get intepretable confusion matrix
        total_per_residue = confusion_matrix_residue.sum(dim=1)
        confusion_matrix_residue = confusion_matrix_residue / total_per_residue
        
        # saving per residue acc in dict 
        per_residue_acc = torch.diag(confusion_matrix_residue).tolist()
        keys = ['I', 'O', 'P', 'S', 'M', 'B']
        per_residue_acc = dict(zip(keys, per_residue_acc))
        average_per_residue_acc = sum(per_residue_acc.values()) / len(per_residue_acc)
        per_residue_acc['average_per_residue_acc'] = average_per_residue_acc
        # handling NaN values
        for key in per_residue_acc.keys():
            if math.isnan(per_residue_acc[key]):
                per_residue_acc[key] = 0.0
      
    if 'class' in result_type:
        # calculating matching overlap accuracy for for each class
        tm_overlap_acc = float(residue_class_matrix[0][0]/(residue_class_matrix[0][0]+residue_class_matrix[1][0]))
        sptm_overlap_acc = float(residue_class_matrix[0][1]/(residue_class_matrix[0][1]+residue_class_matrix[1][1]))
        sp_overlap_acc = float(residue_class_matrix[0][2]/(residue_class_matrix[0][2]+residue_class_matrix[1][2]))
        glob_overlap_acc = float(residue_class_matrix[0][3]/(residue_class_matrix[0][3]+residue_class_matrix[1][3]))
        beta_overlap_acc = float(residue_class_matrix[0][4]/(residue_class_matrix[0][4]+residue_class_matrix[1][4]))

        average_overlap_acc = (tm_overlap_acc+sptm_overlap_acc+sp_overlap_acc+glob_overlap_acc+beta_overlap_acc)/5

        # calculating residue accuracy for each class 
        tm_residue_acc = float(residue_class_matrix[2][0]/(residue_class_matrix[2][0]+residue_class_matrix[3][0]))
        sptm_residue_acc = float(residue_class_matrix[2][1]/(residue_class_matrix[2][1]+residue_class_matrix[3][1]))
        sp_residue_acc = float(residue_class_matrix[2][2]/(residue_class_matrix[2][2]+residue_class_matrix[3][2]))
        glob_residue_acc = float(residue_class_matrix[2][3]/(residue_class_matrix[2][3]+residue_class_matrix[3][3]))
        beta_residue_acc = float(residue_class_matrix[2][4]/(residue_class_matrix[2][4]+residue_class_matrix[3][4]))

        average_residue_acc = (tm_residue_acc+sptm_residue_acc+sp_residue_acc+glob_residue_acc+beta_residue_acc)/5

        # saving the class results in dict
        overlap_values = [average_overlap_acc, tm_overlap_acc, sptm_overlap_acc,
                          sp_overlap_acc, glob_overlap_acc, beta_overlap_acc]
        residue_values = [average_residue_acc, tm_residue_acc, sptm_residue_acc,
                          sp_residue_acc, glob_residue_acc, beta_residue_acc]
        overlap_keys = ['average_overlap_acc', 'tm_overlap_acc', 'sptm_overlap_acc',
                        'sp_overlap_acc', 'glob_overlap_acc', 'beta_overlap_acc']
        residue_keys = ['average_residue_acc', 'tm_residue_acc', 'sptm_residue_acc',
                        'sp_residue_acc', 'glob_residue_acc', 'beta_residue_acc']
        overlap_acc = dict(zip(overlap_keys, overlap_values))
        residue_acc = dict(zip(residue_keys, residue_values))

        # collecting class results in dicts
        #class_results = {'average_overlap_acc': average_overlap_acc,'tm_overlap_acc':tm_overlap_acc,'sptm_overlap_acc':sptm_overlap_acc,'sp_overlap_acc':sp_overlap_acc,'glob_overlap_acc':glob_overlap_acc,
        #                 'beta_overlap_acc':beta_overlap_acc,'average_residue_acc':average_residue_acc,'tm_residue_acc':tm_residue_acc,'sptm_residue_acc':sptm_residue_acc,'sp_residue_acc':sp_residue_acc,
        #                 'glob_residue_acc':glob_residue_acc, 'beta_residue_acc':beta_residue_acc}
        #print(f'Class results: {class_results}')

    # return results 
    if result_type == ['residue']:
        print(f'returning results for residue')
        return per_residue_acc, confusion_matrix_residue
    elif result_type == ['class']: 
        print(f'returning results for class')
        return overlap_acc, residue_acc 
    else:
        print(f'returning results for both')
        return overlap_acc, residue_acc, per_residue_acc, confusion_matrix_residue

def predictions_testset(model, testset, device):
    logits = []
    predictions = []
    labels = []

    with torch.no_grad():
        model.eval()

        for batch in testset :
            batch.to(device)
            batch.num_relation = 2 # temporary 
                        
            # Forward pass
            output, protein_lengths = model(batch)

            # Creating tuples seperating proteins in batch, for output and target labels   
            #output_per_protein = torch.split(output, protein_lengths)
            output_per_protein = output
            targets = labels_to_tensors(batch.label, device)

            # Loop over all proteins in one batch
            for output_, target_ in zip(output_per_protein, targets):
                # Getting predition for one protein
                logit_per_protein = torch.nn.Softmax(dim=1)(output_)
                pred_per_protein = torch.argmax(logit_per_protein, dim=1)

                # Save logit, prediction and label for accuracy evaluation 
                logits.append(logit_per_protein)
                predictions.append(pred_per_protein)
                labels.append(target_)

    return logits, predictions, labels

def weighted_sampling(data_dict, param_dict, protein_list, device):
    
    labels = []
    classes_per_datapoint = []

    for protein in protein_list:
        labels.append(data_dict[protein]['labels'])

    labels = labels_to_tensors(labels, device)

    for tensor in labels:
        classes_per_datapoint.append(type_from_labels(tensor))
    #print(classes_per_datapoint)
    class_sample_count = np.array(
        [len(np.where(classes_per_datapoint == t)[0]) for t in np.unique(classes_per_datapoint)])

    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in classes_per_datapoint])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    
    if(isinstance(param_dict['weight_modulator'],list)):
        #print('List detected')
        for i, weight in enumerate(samples_weight):
            #print(f'Weight: {weight} for class: {classes_per_datapoint[i]}')
            samples_weight[i] = weight * param_dict['weight_modulator'][classes_per_datapoint[i]]
            #print(f'Weight after modulation: {samples_weight[i]}\n')
    return samples_weight 


def add_metrics_to_log(prefix, metrics_dict, log_dict):
    """
    Function to add metrics to log_dict with appropriate keys
    """
    for key, value in metrics_dict.items():
        # Construct the new key by replacing '_acc' with '' and adding the prefix
        new_key = f"{prefix}/{key}"
        log_dict[new_key] = value


def save_performance_tsv(all_logits, all_predictions, all_labels, device):
    print('entered new function ')
    # Getting both performance on class and per residue when testing (not necassarily when training) 
    result_type = ['class', 'residue']
    print(f'hard coded result types = {result_type}')
    protein_acc = bio_acc(all_predictions, all_labels)
    print('Managed to fetch the bio_acc results {protein_acc}')
    print('now entering the bio_top_residue_acc function')
    _, residue_acc, per_residue_acc, confusion_matrix_residue = bio_top_residue_acc(all_predictions, all_labels, device, result_type)


    # calculating AUC for each residue type on all proteins in testsets like one long label and one long protein
    mc_auroc = MulticlassAUROC(num_classes=6, average=None, thresholds=10)
    concat_logits = torch.cat(all_logits, dim=0)
    concat_labels = torch.cat(all_labels, dim=0)
    auc_residue = mc_auroc(concat_logits, concat_labels).tolist()

    # Keys of interest for each dictionary

    residue_keys_of_interest = [
        'tm_residue_acc', 'sptm_residue_acc', 'sp_residue_acc', 
        'glob_residue_acc', 'beta_residue_acc'
    ] 
    protein_keys_of_interest = [
        'tm_top_acc', 'sptm_top_acc', 'sp_top_acc', 'glob_top_acc', 
        'beta_top_acc', 'type_acc', 'tm_type_acc', 'tm_sp_type_acc', 
        'sp_type_acc', 'glob_type_acc', 'beta_type_acc'
    ]
    per_residue_keys_of_interest = [
        'I', 'O', 'P', 'S', 'M', 'B'
    ] 
        
    # Combined keys in desired order for tsv
    combined_keys_of_interest = residue_keys_of_interest + protein_keys_of_interest + ['AUC_' + key for key in per_residue_keys_of_interest] + ['acc_' + key for key in per_residue_keys_of_interest] 

    # Combined valyes in desíred order for tsv, with ',' for excel format (danish) 
    values = [
        str(residue_acc.get(key)).replace('.', ',') for key in residue_keys_of_interest
    ] + [
        str(protein_acc.get(key)).replace('.', ',') for key in protein_keys_of_interest
    ] + [
        str(number).replace('.', ',') for number in auc_residue
    ] + [
        str(per_residue_acc.get(key)).replace('.', ',') for key in per_residue_keys_of_interest
    ]

    return combined_keys_of_interest, values, confusion_matrix_residue



