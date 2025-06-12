from os.path import join

import numpy as np
import pandas as pd
import torch

from deepmreye import pytorch_architecture as architecture
from deepmreye.util import data_generator, util


def train_model(
    dataset,
    generators,
    opts,
    clear_graph=True,
    save=False,
    model_path="./",
    models=None,
    return_untrained=False,
    verbose=0,
):
    """Train the model given a cross validation, \
       hold out or leave one out generator, given model options.

    Parameters
    ----------
    dataset : str
        Description of dataset, used for saving dataset to file
    generators : generator
        Cross validation, Hold out or Leave one out generator, yielding X,y pairs
    opts : dict
        Model options for training
    clear_graph : bool, optional
        If computational graph should be reset before each run, by default True
    save : bool, optional
        If model weights should be saved to file, by default False
    model_path : str, optional
        Filepath to where model weights should be stored, by default './'
    models : torch.nn.Module, optional
        Can be provided if already trained model should be used instead of training a new one, by default None
    return_untrained : bool, optional
        If true, returns untrained but compiled model, by default False
    verbose : int, optional
        Verbosity level, by default 0

    Returns
    -------
    model : torch.nn.Module
        Trained model instance
    model_inference : torch.nn.Module
        Model instance used for inference, provides uncertainty estimate
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if clear_graph:
        torch.cuda.empty_cache()

    # Unpack generators
    (
        training_generator,
        testing_generator,
        single_testing_generators,
        single_testing_names,
        single_training_generators,
        single_training_names,
        full_testing_list,
        full_training_list,
    ) = generators

    # Test datagenerator and get representative X and y
    ((X, y), _) = next(training_generator)
    if verbose > 0:
        print(f"Input shape {X.shape}, Output shape {y.shape}")
        print(
            f"Subjects in training set: {len(single_training_generators)}, "
            f"Subjects in test set: {len(single_testing_generators)}"
        )

    # Get model
    if models is None:
        input_shape = (X.shape[-1], X.shape[1], X.shape[2], X.shape[3])
        model = architecture.StandardModel(input_shape, opts).to(device)
    else:
        model = models
    model_inference = model
    if return_untrained:
        return (model, model_inference)

    optimizer = torch.optim.Adam(model.parameters(), lr=opts["lr"])
    for epoch in range(opts["epochs"]):
        model.train()
        for _ in range(opts["steps_per_epoch"]):
            (batch, _) = next(training_generator)
            X_batch, y_batch = batch
            X_batch = torch.from_numpy(X_batch).permute(0, 4, 1, 2, 3).float().to(device)
            y_batch = torch.from_numpy(y_batch).float().to(device)

            optimizer.zero_grad()
            pred, conf = model(X_batch)
            loss_euc, loss_conf = architecture.compute_standard_loss(conf, y_batch, pred)
            loss = opts["loss_euclidean"] * loss_euc + opts["loss_confidence"] * loss_conf
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            (val_batch, _) = next(testing_generator)
            X_val, y_val = val_batch
            X_val = torch.from_numpy(X_val).permute(0, 4, 1, 2, 3).float().to(device)
            y_val = torch.from_numpy(y_val).float().to(device)
            pred, conf = model(X_val)
            loss_euc, loss_conf = architecture.compute_standard_loss(conf, y_val, pred)
        if verbose > 0:
            print(
                f"Epoch {epoch + 1}/{opts['epochs']} - "
                f"train loss {loss.item():.4f} - val euclidean {loss_euc.item():.4f}"
            )

    # Save model weights
    if save:
        torch.save(model.state_dict(), join(model_path, f"modelinference_{dataset}.pth"))

    return (model, model_inference)


def evaluate_model(dataset, model, generators, save=False, model_path="./", model_description="", verbose=0, **args):
    """Evaluate model performance given model and generators \
       used for training the model.

       Evaluate only on test set.

    Parameters
    ----------
    dataset : str
        Description of dataset, used for saving dataset to file
    model : torch.nn.Module
        Trained model instance
    generators : generator
        Cross validation, Hold out or Leave one out generator, yielding X,y pairs
    save : bool, optional
        If true, save test set predictions to file, by default False
    model_path : str, optional
        Filepath to where model weights should be stored, by default './'
    model_description : str, optional
        Description of model used for saving the model evaluations, by default ''
    verbose : int, optional
        Verbosity level, by default 0

    Returns
    -------
    evaluation: dict
        Raw gaze coordinates, returned for each participant
    scores: pandas DataFrame
        Evaluation metrics for gaze coordinates (Pearson, R2-Score, Euclidean Error)
    """
    (
        training_generator,
        testing_generator,
        single_testing_generators,
        single_testing_names,
        single_training_generators,
        single_training_names,
        full_testing_list,
        full_training_list,
    ) = generators
    evaluation, scores = dict(), dict()
    device = next(model.parameters()).device
    for idx, subj in enumerate(full_testing_list):
        X, real_y = data_generator.get_all_subject_data(subj)
        X_t = torch.from_numpy(X).permute(0, 4, 1, 2, 3).float().to(device)
        with torch.no_grad():
            pred_y_t, conf_t = model(X_t)
        pred_y = pred_y_t.cpu().numpy()
        euc_pred = conf_t.cpu().numpy()
        evaluation[subj] = {"real_y": real_y, "pred_y": pred_y, "euc_pred": euc_pred}

        # Quantify predictions
        df_scores = util.get_model_scores(real_y, pred_y, euc_pred, **args)
        scores[subj] = df_scores

        # Print evaluation
        if verbose > 0:
            print(
                f"{util.color.BOLD}{idx + 1} / {len(single_testing_names)} - "
                f"Model Performance for {subj}{util.color.END}"
            )
            if verbose > 1:
                pd.set_option("display.width", 120)
                pd.options.display.float_format = "{:.3f}".format  # noqa
                print(df_scores)
            else:
                print(
                    f"Default: r={df_scores[('Pearson', 'Mean')]['Default']:.3f}, "
                    f"subTR: r={df_scores[('Pearson', 'Mean')]['Default subTR']:.3f}, "
                    f"Euclidean Error: {df_scores[('Eucl. Error', 'Mean')]['Default']:.3f}°"
                )
            print("\n")

    # Save dict
    if save:
        np.save(join(model_path, f"results{model_description}_{dataset}.npy"), evaluation)

    return (evaluation, scores)
