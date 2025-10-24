import os
import torch
import pandas as pd
from bikebench.benchmarking.scoring import construct_scorer, MainScores, DetailedScores
from bikebench.design_evaluation.design_evaluation import get_standard_evaluations, construct_tensor_evaluator

from bikebench.conditioning import conditioning
from tqdm import trange, tqdm
from bikebench.transformation import ordered_columns

def get_train_conditions(n, randomize=True, device="cpu", mode = "embedding"):
    rider_conditions = conditioning.sample_riders(n, split="train", randomize=randomize, device=device)
    use_case_conditions = conditioning.sample_use_case(n, split="train", randomize=randomize, device=device)

    if mode == "text":
        texts = conditioning.sample_text(n, split="train", randomize=randomize)
        conditions = {"Rider": rider_conditions, "Use Case": use_case_conditions, "Text": texts}
        return conditions
    elif mode == "embedding":
        image_embeddings = conditioning.sample_embedding(n, split="train", randomize=randomize, device=device)
        conditions = {"Rider": rider_conditions, "Use Case": use_case_conditions, "Embedding": image_embeddings}
        return conditions

def get_test_conditions(device="cpu"):
    rider_conditions = conditioning.sample_riders(100, split="test", device=device)
    use_case_conditions = conditioning.sample_use_case(100, split="test", device=device)
    image_embeddings = conditioning.sample_embedding(100, split="test", device=device)

    rider_conditions_repeated = rider_conditions.repeat_interleave(100, dim=0)
    use_case_conditions_repeated = use_case_conditions.repeat_interleave(100, dim=0)
    image_embeddings_repeated = image_embeddings.repeat_interleave(100, dim=0)

    conditions = {"Rider": rider_conditions_repeated, "Use Case": use_case_conditions_repeated, "Embedding": image_embeddings_repeated}

    return conditions

def get_single_test_condition(idx=0, device="cpu", mode = "embedding"):
    rider_condition = conditioning.sample_riders(100, split="test", device=device)
    use_case_condition = conditioning.sample_use_case(100, split="test", device=device)
    rider_condition = rider_condition[idx].to(device)
    use_case_condition = use_case_condition[idx].to(device)
    if mode == "text":
        text = conditioning.sample_text(100, split="test")
        text = text[idx]
        condition = {"Rider": rider_condition, "Use Case": use_case_condition, "Text": text}
        return condition

    elif mode == "embedding":
        image_embedding = conditioning.sample_embedding(100, split="test", device=device)


        image_embedding = image_embedding[idx].to(device)

        condition = {"Rider": rider_condition, "Use Case": use_case_condition, "Embedding": image_embedding}
        return condition
    else:
        raise ValueError("mode must be 'text' or 'embedding'")

def evaluate_designs(result_tens, evaluate_as_aggregate = False):
    data_columns = ordered_columns.bike_bench_columns
    evaluations = get_standard_evaluations("cpu")
    evaluator, requirement_names, is_objective, is_conditional = construct_tensor_evaluator(evaluations, data_columns, device="cpu")

    if evaluate_as_aggregate:
        condition = get_test_conditions("cpu")
        main_scorer = construct_scorer(MainScores, evaluations, data_columns, "cpu")
        detailed_scorer = construct_scorer(DetailedScores, evaluations, data_columns, "cpu")

        main_scores = main_scorer(result_tens, condition)
        detailed_scores = detailed_scorer(result_tens, condition)
    else:
        all_main_scores = []
        all_detailed_scores = []
        all_evaluation_scores = []

        main_scorer = construct_scorer(MainScores, evaluations, data_columns, "cpu")
        detailed_scorer = construct_scorer(DetailedScores, evaluations, data_columns, "cpu")
        for i in trange(100):
            result_slice = result_tens[i*100:(i+1)*100]
            condition = get_single_test_condition(i, "cpu")

            result_slice = result_slice.detach().cpu()
            evaluation_scores = evaluator(result_slice, condition)

            main_scores = main_scorer(result_slice, condition, preevaluated_scores = evaluation_scores) #main scores is a series
            detailed_scores = detailed_scorer(result_slice, condition, preevaluated_scores = evaluation_scores) #detailed scores is a series

            all_main_scores.append(main_scores)
            all_detailed_scores.append(detailed_scores)
            all_evaluation_scores.append(evaluation_scores)
        main_scores = pd.concat(all_main_scores, axis=1).T
        detailed_scores = pd.concat(all_detailed_scores, axis=1).T
        all_evaluation_scores = torch.stack(all_evaluation_scores)
        main_scores = main_scores.mean()
        detailed_scores = detailed_scores.mean()

    return main_scores, detailed_scores, all_evaluation_scores


        




# def get_condition_by_idx(idx=0):
#     rider_condition = conditioning.sample_riders(10, split="test")
#     use_case_condition = conditioning.sample_use_case(10, split="test")
#     image_embeddings = conditioning.sample_image_embedding(10, split="test")
#     condition = {"Rider": rider_condition[idx], "Use Case": use_case_condition[idx], "Embedding": image_embeddings[idx]}
#     return condition

# def get_conditions_10k():
#     rider_condition = conditioning.sample_riders(10000, split="test")
#     use_case_condition = conditioning.sample_use_case(10000, split="test")
#     image_embeddings = conditioning.sample_image_embedding(10000, split="test")
#     conditions = {"Rider": rider_condition, "Use Case": use_case_condition, "Embedding": image_embeddings}
#     return conditions

# def evaluate_uncond(result_tens, name, cond_idx, data_columns, device, save=True):

#     condition = get_condition_by_idx(cond_idx)
    
#     main_scorer = construct_scorer(MainScores, get_standard_evaluations(device), data_columns)
#     detailed_scorer = construct_scorer(DetailedScores, get_standard_evaluations(device), data_columns)

#     main_scores = main_scorer(result_tens, condition)
    
#     detailed_scores = detailed_scorer(result_tens, condition)
    
#     if save:
#         result_tens = result_tens.cpu()
#         torch.save(result_tens, os.path.join(result_dir, "result_tens.pt"))
#         main_scores.to_csv(os.path.join(result_dir, "main_scores.csv"), index_label=False, header=False)
#         detailed_scores.to_csv(os.path.join(result_dir, "detailed_scores.csv"), index_label=False, header=False)
#     return main_scores, detailed_scores

# def evaluate_cond(result_tens, name, data_columns, device, save=True):
#     condition = get_conditions_10k()

#     condition = {"Rider": condition["Rider"], "Use Case": condition["Use Case"], "Embedding": condition["Embedding"]}

#     result_dir = os.path.join("results", "conditional", name)
#     os.makedirs(result_dir, exist_ok=True)

#     main_scorer = construct_scorer(MainScores, get_standard_evaluations(device), data_columns, device)
#     detailed_scorer = construct_scorer(DetailedScores, get_standard_evaluations(device), data_columns, device)

#     main_scores = main_scorer(result_tens, condition)
#     detailed_scores = detailed_scorer(result_tens, condition)

#     if save:
#         result_tens = result_tens.cpu()
#         torch.save(result_tens, os.path.join(result_dir, "result_tens.pt"))
#         main_scores.to_csv(os.path.join(result_dir, "main_scores.csv"), index_label=False, header=False)
#         detailed_scores.to_csv(os.path.join(result_dir, "detailed_scores.csv"), index_label=False, header=False)

#     return main_scores, detailed_scores


# def create_score_report_conditional():
#     """
#     Looks through the results folder and creates a score report for each conditional result.
#     """
#     all_scores = []
#     result_dir = os.path.join("results", "conditional")
#     for name in os.listdir(result_dir):
#         if os.path.isdir(os.path.join(result_dir, name)):
#             main_scores = pd.read_csv(os.path.join(result_dir, name, "main_scores.csv"), header=None)
#             main_scores.columns = ["Metric", "Score"]
#             main_scores["Model"] = name
#             all_scores.append(main_scores)
#     all_scores = pd.concat(all_scores, axis=0)
#     #make metric names the three columns, make models the rows
#     all_scores = all_scores.pivot(index="Model", columns="Metric", values="Score")
#     #drop the index name and the column name
#     all_scores.columns.name = None
#     all_scores.index.name = None
    
#     return all_scores

# def create_score_report_unconditional():
#     """
#     Looks through the results folder and creates a score report for each unconditional result.
#     """
#     all_scores = []
#     result_dir = os.path.join("results", "unconditional")
#     for i in range(10):
#         c_dir = os.path.join(result_dir, f"cond_{i}")
#         for name in os.listdir(c_dir):
#             dirname = os.path.join(c_dir, name)
#             if os.path.isdir(dirname):
#                 main_scores = pd.read_csv(os.path.join(dirname, "main_scores.csv"), header=None)
#                 main_scores.columns = ["Metric", "Score"]
#                 main_scores["Model"] = name
#                 main_scores["Condition"] = i
#                 all_scores.append(main_scores)
#     all_scores = pd.concat(all_scores, axis=0)
#     #average over condition 
#     all_scores = all_scores.groupby(["Model", "Metric"]).mean().reset_index()
#     #make metric names the three columns, make models the rows
#     all_scores = all_scores.pivot(index="Model", columns="Metric", values="Score")
#     #drop the index name and the column name
#     all_scores.columns.name = None
#     all_scores.index.name = None
#     return all_scores


# def rescore_unconditional(data_columns, device, cond_idxs = None, model_names = None, results_root="results/unconditional"):
#     """
#     Recompute main and detailed scores for all unconditional results.
#     Overwrites only the CSV score files, leaves result_tens.pt untouched.
#     """

#     evals = get_standard_evaluations(device)
#     main_scorer     = construct_scorer(MainScores,    evals, data_columns, device)
#     detailed_scorer = construct_scorer(DetailedScores, evals, data_columns, device)
#     device = torch.device(device)
#     if cond_idxs is None:
#         cond_idxs = range(10)
#     for cond_idx in tqdm(cond_idxs):
#         cond_dir = os.path.join(results_root, f"cond_{cond_idx}")
#         if not os.path.isdir(cond_dir):
#             continue
#         # fetch the one shared condition for this index
#         condition = get_condition_by_idx(cond_idx)

#         if model_names is not None:
#             models = model_names
#         else:
#             models = os.listdir(cond_dir)

#         for model_name in models:
#             model_dir = os.path.join(cond_dir, model_name)
#             tensor_path = os.path.join(model_dir, "result_tens.pt")
#             if not os.path.isdir(model_dir) or not os.path.isfile(tensor_path):
#                 continue

#             # load results
#             result_tens = torch.load(tensor_path, map_location=device)

#             # rescore
#             main_scores     = main_scorer(result_tens, condition)
#             detailed_scores = detailed_scorer(result_tens, condition)

#             # overwrite only the CSVs
#             main_scores.to_csv(
#                 os.path.join(model_dir, "main_scores.csv"), header=False
#             )
#             detailed_scores.to_csv(
#                 os.path.join(model_dir, "detailed_scores.csv"), header=False
#             )


# def rescore_conditional(data_columns, device, model_names, results_root="results/conditional"):
#     """
#     Recompute main and detailed scores for all conditional results.
#     Overwrites only the CSV score files, leaves result_tens.pt untouched.
#     """
#     device = torch.device(device)
#     # fetch the full 10k‐point condition set once
#     condition = get_conditions_10k()

#     # build scorers
#     evals = get_standard_evaluations(device)
#     main_scorer     = construct_scorer(MainScores, evals, data_columns, device)
#     detailed_scorer = construct_scorer(DetailedScores, evals, data_columns, device)


#     if model_names is not None:
#         models = model_names
#     else:
#         models = os.listdir(results_root)
#     for model_name in models:
#         model_dir  = os.path.join(results_root, model_name)
#         tensor_path = os.path.join(model_dir, "result_tens.pt")
#         if not os.path.isdir(model_dir) or not os.path.isfile(tensor_path):
#             continue

#         # load results
#         result_tens = torch.load(tensor_path, map_location=device)

#         # rescore
#         main_scores     = main_scorer(result_tens, condition)
#         detailed_scores = detailed_scorer(result_tens, condition)

#         # overwrite only the CSVs
#         main_scores.to_csv(
#             os.path.join(model_dir, "main_scores.csv"), header=False
#         )
#         detailed_scores.to_csv(
#             os.path.join(model_dir, "detailed_scores.csv"), header=False
#         )