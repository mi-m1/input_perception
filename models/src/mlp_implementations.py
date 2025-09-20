import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import random

import os

class LargerNN(nn.Module):
    def __init__(self, input_size, dropout_rate=0.3):
        super(LargerNN, self).__init__()

        self.fc1 = nn.Linear(input_size, 1024)
        self.dropout = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(1024, 1024)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(1024, 1024)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.out = nn.Linear(1024, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)

        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)

        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)

        x = torch.sigmoid(self.out(x))
        return x


class QuevedoNN(nn.Module):  # If this is not a spiking model, this name is clearer.
    def __init__(self, input_size):
        super(QuevedoNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)       # Input layer → Hidden layer 1
        self.fc2 = nn.Linear(512, 512)     # Hidden layer 1 → Hidden layer 2
        self.out = nn.Linear(512, 1)       # Hidden layer 2 → Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))            # Apply ReLU after first layer
        x = F.relu(self.fc2(x))            # Apply ReLU after second layer
        x = torch.sigmoid(self.out(x))     # Apply Sigmoid for binary classification
        return x


def set_seed(seed=42):
    random.seed(seed)                # Python built-in random module
    np.random.seed(seed)             # NumPy
    torch.manual_seed(seed)          # PyTorch CPU
    torch.cuda.manual_seed(seed)     # PyTorch CUDA (if using GPU)
    torch.cuda.manual_seed_all(seed) # All GPUs
    torch.backends.cudnn.deterministic = True  # For reproducibility
    torch.backends.cudnn.benchmark = False     # Turn off auto-optimization


def align(df_feats, df_isd, col_feats, col_isd):
    df_merged = df_isd.merge(df_feats, left_on=col_isd, right_on=col_feats, how="left")
    return df_merged

def run_mlp_model(df_merged, feature_cols, col_feats, model_name, target_col='error'):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    indicies = np.arange(len(df_merged))
    X = df_merged[feature_cols].values
    y = df_merged['error'].values

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)


    # Train-test split
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(X, y, indicies, test_size=0.2, random_state=42, stratify=y)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    # Create DataLoader
    batch_size = 32

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(X_val, y_val)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model, loss, optimizer
    if model_name == "larger":
        model = LargerNN(input_size=X_train.shape[1])
    elif model_name == "quevedo":
        model = QuevedoNN(input_size=X_train.shape[1])


    model = model.to(device)  # Move model to GPU if available

    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # Move data to GPU if available
            
            optimizer.zero_grad()
            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * X_batch.size(0)

            # Compute accuracy
            predictions = (outputs >= 0.5).float()
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)

        epoch_loss /= total
        accuracy = correct / total * 100
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")


    from sklearn.metrics import precision_score, recall_score, f1_score

    # X_val = X_val.to(device)  # Move validation data to GPU if available
    # #Get all predictions and labels
    # model.eval()
    # with torch.no_grad():
    #     outputs = model(X_val)
    #     predictions = (outputs >= 0.5).float()

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            probs = outputs.cpu().numpy()

            predictions = (outputs >= 0.5).float()

            all_probs.append(probs)
            all_preds.append(predictions.cpu())
            all_labels.append(y_batch.cpu())
    
    y_probs = np.vstack(all_probs)
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    #complementary results
    complementary_results = pd.DataFrame({
        "y_true": y_true.flatten(),
        "y_pred":y_pred.flatten(),
        "y_prob":y_probs.flatten(),
    })

    # complementary_results["df_index"] = idx_val
    # complementary_results = complementary_results.set_index("df_index")

    # results_inspection_df = df_merged.loc[complementary_results.index].copy()
    # results_inspection_df = results_inspection_df.join(complementary_results)

    # print(f"results_inspection_df:\n{results_inspection_df}")

    # probs_for_1 = complementary_results[complementary_results["y_true"] == 1.0]["y_prob"]
    # print(f"probs_for_1:\n{probs_for_1}")
    
    # # 1. Filter: correct prediction of class 1
    # correct_class_1 = complementary_results[
    #     (complementary_results["y_true"] == 1.0) &
    #     (complementary_results["y_pred"] == 1.0)
    # ]

    # # 2. Find the corresponding rows in df_merged
    # correct_rows = df_merged.loc[correct_class_1.index].copy()

    # # 3. Now filter those where gold == "i"
    # final_examples = correct_rows[
    #     (correct_rows["gold"] == "i") &
    #     (correct_rows["error"] == 1.0)]

    # # 4. Add prediction probability from complementary_results
    # final_examples["y_prob"] = correct_class_1.loc[final_examples.index, "y_prob"]

    # # 5. Display examples
    # for i, row in final_examples.head(10).iterrows():
    #     print(f"Index: {i}")
    #     print(f"Gold label: {row['gold']}")
    #     print(f"Sentence: {row['sentence_cleaned']}")
    #     print(f"Predicted: 1.0 | True: 1.0")
    #     print(f"Probability (class=1): {row['y_prob']:.4f}")
    #     print(f"Probability (class=0): {1 - row['y_prob']:.4f}")
    #     print("-" * 60)

    # Convert to NumPy
    # y_true = y_val.cpu().numpy()
    # y_pred = predictions.cpu().numpy()

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")

    # y_true = y_val.cpu().numpy().flatten()
    # y_pred = (model(X_val).detach().cpu().numpy() >= 0.5).astype(int).flatten()

    # Compute per-class F1 score
    f1_per_class = f1_score(y_true, y_pred, average=None)

    print(f"F1 Score for class 0: {f1_per_class[0]:.2f}")
    print(f"F1 Score for class 1: {f1_per_class[1]:.2f}")

    return f1_per_class * 100


tasks = [
    "dice",
    "mohx",
    "trofi",
    "pub_14_metonymy",
    "conmec",
]

dict_company_models = {
    "meta-llama": [
        "Llama-3.1-8B-Instruct",
        "Llama-3.2-3B-Instruct",
        "Llama-3.2-1B-Instruct",
        ],
    "Qwen": [
        "Qwen2-1.5B-Instruct",
        "Qwen2.5-0.5B-Instruct",
        "Qwen2.5-7B-Instruct-1M", 
        "Qwen2.5-14B-Instruct-1M",   
        ],}

# Set the seed for reproducibility
set_seed(42)

def run_pipeline(features_list, mlp_model, output_dir, remarks, baseline=False, save=False):

    results_dict = {}
    gamma = 0.5
    
    # choose tasks based on remarks
    if "dice_specific" in remarks:
        selected_tasks = ["dice"]
    elif "pub14metonymy_specific" in remarks:
        selected_tasks = ["pub_14_metonymy"]
    elif "rest_specific" in remarks:
        selected_tasks = ["conmec", "trofi", "mohx"]
    else:
        selected_tasks = tasks  

    for task in selected_tasks:
        for company in dict_company_models.keys():
            for model in dict_company_models[company]:

                print(f"task:{task} model:{model}")
                if baseline:
                    df_feats = pd.read_csv(f'/mnt/parscratch/users/acq22zm/surprisal/baselines/features_sentenceLevel/baseline_features/{task}/{company}/{model}.csv')
                else:
                    df_feats = pd.read_csv(f'/mnt/parscratch/users/acq22zm/surprisal/experiments/correlation/features/{task}/{company}/{model}/cws_y{gamma}.csv')
                
                
                if task == "dice":
                    isd_file_path = f"/mnt/parscratch/users/acq22zm/surprisal/scripts/postprocessing_lighteval_evaluations/{task}_rearranged/{company}/{model}.csv"
                else:
                    isd_file_path = f"/mnt/parscratch/users/acq22zm/surprisal/experiments/prompt_evaluation/{task}/postprocessed/{company}/{model}_accuracy.csv"
                                

                df_isd = pd.read_csv(isd_file_path)
                # print(df_isd["accuracy_per_sample"])
                df_isd['error'] = 1.0 - df_isd['accuracy_per_sample']

                if baseline:
                    if task == "mohx" or task == "trofi":
                        col_isd = "sentence"
                        col_feats = "sentence_cleaned"
                    elif task == "pub_14_metonymy":
                        col_isd = "id"
                        col_feats = "base_expression"
                    elif task == "conmec":
                        col_isd = "Sentence"
                        col_feats = "Sentence"
                    elif task == "dice":
                        col_isd = "sentence_cleaned"
                        col_feats = "sentence_cleaned"
                else:

                    if task == "dice":
                        col_isd = "sentence_cleaned"
                        col_feats = "sentence"
                    elif task == "mohx" or task == "trofi":
                        col_isd = "sentence"
                        col_feats = "sentence"
                    elif task == "conmec":
                        col_isd = "Sentence"
                        col_feats = "sentence"
                    elif task == "pub_14_metonymy":
                        col_isd = "id"
                        col_feats = "id"

                print(f"df_isd columns: {df_isd.columns}")
                print(f"df_feats columns: {df_feats.columns}")
                print(f"col_isd:{col_isd}, col_feats:{col_feats}")

                df_merged = align(df_feats, df_isd, col_feats, col_isd)
                print(f"df_merged:\n{df_merged}")
                print(f"df_merged (shape):{df_merged.shape}")

                #rows to drop
                rows_to_drop = df_merged[df_merged[features_list].isna().any(axis=1)]
                print(f"rows_to_drop:\n{rows_to_drop}")

                df_merged = df_merged.dropna(subset=features_list)
                print(f"cleaned df_merged:{df_merged.shape}")


                f1_results_both_classes = run_mlp_model(
                                df_merged,
                                features_list,
                                col_feats,
                                model_name=mlp_model,
                                target_col='error',
                            )
                
                f1_class_1 = f1_results_both_classes[1]         

                if model not in results_dict:
                    results_dict[model] = {}

                results_dict[model][task] = f1_class_1

                print("-"*100)

    results_df = pd.DataFrame.from_dict(results_dict, orient='index')

    print(results_df)

    if save:
        os.makedirs(f"{output_dir}/", exist_ok=True)
        results_df.to_csv(f"{output_dir}/{remarks}.csv", index=True)
    else:
        pass

if __name__ == "__main__":
    def process_options():
        import argparse
        parser = argparse.ArgumentParser(description="Run MLP pipeline with specified parameters.")
        parser.add_argument("--features",nargs="*", type=str, required=True, help="Features to use separated by space and pass in nothing to use all.")
        parser.add_argument("--model", type=str, required=True, help="Model name (larger, quevedo).")
        parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results.")
        parser.add_argument("--remarks", type=str, required=True, help="Remarks for the output file.")
        parser.add_argument("--baseline", type=str, help="Use baseline features.")
        parser.add_argument("--save", action='store_true', help="Save the results.")
        return parser.parse_args()


    args = process_options()

    run_pipeline(
        features_list=args.features,
        mlp_model=args.model,
        output_dir=args.output_dir,
        remarks=args.remarks,
        baseline=args.baseline,
        save=args.save
    )