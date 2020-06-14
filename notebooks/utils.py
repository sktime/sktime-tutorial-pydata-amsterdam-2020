#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = ["load_pressure", "load_temperature", "load_experiments"]

import pandas as pd


def _load_data():
    """Helper function to load data for tutorial

    References
    ----------
    Data is a small extract from the Tennessee Eastman Process Simulation Data
    for Anomaly Detection. The full data set is available at
    https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN
    /6C3JR1
    """
    return pd.read_csv("../data/chemical_process_data.csv", index_col=0,
                       header=[0, 1])


def _load_single_series(name):
    data = _load_data()
    series = data.loc[1, name].reset_index(drop=True)
    series.name = name
    return series


def load_experiments(variables=None):
    data = _load_data()
    if variables is not None:
        return data.loc[:, variables]
    return data


def load_pressure():
    return _load_single_series("pressure")


def load_temperature():
    return _load_single_series("temperature")
