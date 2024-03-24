import pickle 
import json
import csv
import numpy as np
#from torch import tensor
#from torchmetrics.classification import MulticlassAUROC
from Utils import *
from Models import *
from Dataset import * 
from torch_geometric.loader import DataLoader
from proteinworkshop.features.factory import ProteinFeaturiser  

def test(args):

    # Load in dict with parameters for current run 
    with open(f'{args.run}parameters.json', 'r') as file:
        param = json.load(file)

    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        device = torch.device('cuda')
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print(f'GPU: {torch.cuda.get_device_name(0)} will be used')

    # If GPU is not available
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device('cpu')

    # Initialize featuriser for data 
    featuriser = ProteinFeaturiser(
        representation = param['representation'],
        scalar_node_features = param['scalar_node_features'],
        vector_node_features=[],
        edge_types = param['edge_types'],
        scalar_edge_features = param['scalar_edge_features'],
        vector_edge_features=[]
        )

    # Load in model
    model = GraphEncDec(featuriser=featuriser,hidden_dim_GCN=param['hidden_dim_GCN'], decoder_type=param['decoder_type'], LSTM_hidden_dim=param['LSTM_hidden_dim'], dropout=param['dropout'], type=param['type'],LSTMnormalization=param['LSTMnormalization'] ,lstm_layers=param['lstm_layers'], encoder_concat_hidden=param['encoder_concat_hidden'], num_relation=len(param['edge_types']), input_dim=param['input_dim'])
    model.to(device)

    # Load in dict with information about each protein 
    with open(param['data_dir'] + 'data_partitions.pkl', 'rb') as file:
        data_dict = pickle.load(file)

    # Test datasplits 
    #data_splits =[[3],[4],[0],[1],[2]]
    data_splits =[[3]]

    # Lists for predictions and labels for entire dataset
    all_logits, all_predictions, all_labels = [], [], []

    k = 0
    for split in data_splits:
        k += 1
        print(f'Collecting results for model {k}')
        # Load in model for specific CV fold 
        #model.load_state_dict(torch.load(param['model_dir'] + f'model.{k}.pt'))
        model.load_state_dict(torch.load(f'{args.run}best_model.cv{k}.pt'))
       
        # Are not present in DeepTMHMMs cross validated results
        ids_to_remove = ['P02930', 'A1JUB7']

        # Defining the proteins list 
        test_protein_list = cv_fold_split(data_dict, split)
        
        # Remove the two missing proteins IDs
        test_protein_list = [id for id in test_protein_list if id not in ids_to_remove]
        testDset = GearNetDataset(root = param['data_dir'], protein_list = test_protein_list, data_dict = data_dict) 
        test_loader = DataLoader(testDset, batch_size = param['batch_size'], shuffle = False,num_workers = 0)

        logits, predictions, labels = predictions_testset(model, test_loader, device)

        # Saving predictions and labels for the current testset
        all_logits.append(logits) 
        all_predictions.append(predictions)
        all_labels.append(labels)

    print(f'Combining all models, to get overall performance')

    # Combine all predictions and labels into one list
    all_logits = [item for sublist in all_logits for item in sublist]
    all_predictions = [item for sublist in all_predictions for item in sublist]
    all_labels = [item for sublist in all_labels for item in sublist]
    
    print('finished combining all logits, predictions and labels, now entering new functions')
    # Evaluating performance of models from cv

    names_of_measures, values, confusion_matrix_residue = save_performance_tsv(all_logits, all_predictions, all_labels, device)
    """
    # calculating AUC for each residue type on all proteins in testsets like one long label and one long protein
    mc_auroc = MulticlassAUROC(num_classes=6, average=None, thresholds=10)
    concat_logits = torch.cat(all_logits, dim=0)
    concat_labels = torch.cat(all_labels, dim=0)
    auc_residue = mc_auroc(concat_logits, concat_labels).tolist()
    #TEMPORARY PARAMETER, ADD TO PARAM LATER
    result_type = ['residue','class']
    residue_results = bio_top_residue_acc(all_predictions, all_labels, device, result_type)
    protein_results = bio_acc(all_predictions, all_labels)
    
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
    #residue_keys_of_interest = ['tm_residue_acc', 'sptm_residue_acc', 'sp_residue_acc', 'glob_residue_acc', 'beta_residue_acc']
    #protein_keys_of_interest = ['tm_top_acc', 'sptm_top_acc', 'sp_top_acc', 'glob_top_acc', 'beta_top_acc', 'type_acc', 'tm_type_acc', 'tm_sp_type_acc', 'sp_type_acc', 'glob_type_acc', 'beta_type_acc']

    # Combined keys in desired order for CSV
    combined_keys_of_interest = residue_keys_of_interest + protein_keys_of_interest

    # Extracting values (per residue) 
    acc_residue = residue_results['per_residue_acc']
    #acc_residue = [str(number).replace('.', ',') for number in acc_residue]
    #auc_residue = auc_residue.tolist()
    #auc_residue = [str(number).replace('.', ',') for number in auc_residue]

    print(acc_residue)
    print(auc_residue)
    # Extracing values (per class) and combining together 
    values = [
        str(residue_results.get(key)).replace('.', ',') for key in residue_keys_of_interest
    ] + [
        str(protein_results.get(key)).replace('.', ',') for key in protein_keys_of_interest
    ] + [
        str(number).replace('.', ',') for number in auc_residue
    ] + [
        str(number).replace('.', ',') for number in acc_residue]
    
    # Write the values to a CSV file
    with open(f'{args.run}performace.tsv', 'w', newline='') as tsvfile:
        tsvwriter = csv.writer(tsvfile, delimiter=' ')
        tsvwriter.writerow(combined_keys_of_interest)
        tsvwriter.writerow(values)
        #tsvwriter.writerow([str(number).replace('.', ',') for number in acc_residue])
        #auc_residue = auc_residue.tolist()
        #tsvwriter.writerow([str(number).replace('.', ',') for number in auc_residue])
        cmr = residue_results['confusion_matrix_residue'].tolist()
        cmr = [['{:.3f}'.format(item) for item in sublist] for sublist in cmr]
        for row in cmr:
            tsvwriter.writerow(row)
    """
    # Write the values to a tsv  file
    with open(f'{args.run}performace.tsv', 'w', newline='') as tsvfile:
        tsvwriter = csv.writer(tsvfile, delimiter=' ')
        tsvwriter.writerow(names_of_measures)
        tsvwriter.writerow(values)
        cmr = confusion_matrix_residue.tolist()
        cmr = [['{:.3f}'.format(item) for item in sublist] for sublist in cmr]
        for row in cmr:
            tsvwriter.writerow(row)

    print(f'Results saved as performance.tsv in {args.run}')

    """
    # Evaluating performance of the 5 models
    bio_results = bio_acc(all_predictions, all_labels)
    keys_of_interest = ['tm_top_acc', 'sptm_top_acc', 'sp_top_acc', 'glob_top_acc', 'beta_top_acc', 'type_acc', 'tm_type_acc', 'tm_sp_type_acc', 'sp_type_acc', 'glob_type_acc', 'beta_type_acc']
    bio_results = {key: bio_results[key] for key in keys_of_interest}
       
    residue_results = bio_top_residue_acc(all_predictions, all_labels, device)
    keys_of_interest = ['tm_residue_acc', 'sptm_residue_acc', 'sp_residue_acc', 'glob_residue_acc', 'beta_residue_acc']
    residue_results = {key: residue_results[key] for key in keys_of_interest}

    with open(f'/zhome/be/1/138857/special_project/results/results{args.model_num}.txt', 'w') as file:
        # Write key-value pairs from residue_results
        for key, value in residue_results.items():
            file.write(f'{key}: {value}\n') 
        # Write key-value pairs from bio_results
        for key, value in bio_results.items():
            file.write(f'{key}: {value}\n')
    print(f'Results saved as results{args.model_num}.txt')
    """
