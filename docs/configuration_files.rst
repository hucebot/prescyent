########################
API Configurations Files
########################

.. toctree::
   :maxdepth: 1

*******
Dataset
*******


**TrajectoriesDatasetConfig**
=============================
.. autoclass:: prescyent.dataset.config.TrajectoriesDatasetConfig
   :members:
   :exclude-members: model_config, model_extra, model_fields, check_context_keys
   :undoc-members:
   :inherited-members:
   :show-inheritance:

----

**H36MDatasetConfig**
=====================

.. autoclass:: prescyent.dataset.datasets.human36m.config.H36MDatasetConfig
   :members:
   :exclude-members: model_config, model_extra, model_fields, check_context_keys
   :undoc-members:
   :inherited-members:
   :show-inheritance:

----

**AndyDatasetConfig**
=====================

.. autoclass:: prescyent.dataset.datasets.andydataset.config.AndyDatasetConfig
   :members:
   :exclude-members: model_config, model_extra, model_fields, check_context_keys
   :undoc-members:
   :inherited-members:
   :show-inheritance:

----

**TeleopIcubDatasetConfig**
===========================

.. autoclass:: prescyent.dataset.datasets.teleop_icub.config.TeleopIcubDatasetConfig
   :members:
   :exclude-members: model_config, model_extra, model_fields, check_context_keys
   :undoc-members:
   :inherited-members:
   :show-inheritance:

----

**SCCDatasetConfig**
====================

.. autoclass:: prescyent.dataset.datasets.synthetic_circle_clusters.config.SCCDatasetConfig
   :members:
   :exclude-members: model_config, model_extra, model_fields, check_context_keys
   :undoc-members:
   :inherited-members:
   :show-inheritance:

----

**SSTDatasetConfig**
====================

.. autoclass:: prescyent.dataset.datasets.synthetic_simple_trajs.config.SSTDatasetConfig
   :members:
   :exclude-members: model_config, model_extra, model_fields, check_context_keys
   :undoc-members:
   :inherited-members:
   :show-inheritance:

******
Scaler
******


**ScalerConfig**
================

.. autoclass:: prescyent.scaler.config.ScalerConfig
   :members:
   :exclude-members: model_config, model_extra, model_fields
   :undoc-members:
   :inherited-members:
   :show-inheritance:

*********
Predictor
*********


**PredictorConfig**
===================

.. autoclass:: prescyent.predictor.config.PredictorConfig
   :members:
   :exclude-members: model_config, model_extra, model_fields
   :undoc-members:
   :inherited-members:
   :show-inheritance:

----

**PrompConfig**
===============

.. autoclass:: prescyent.predictor.promp.config.PrompConfig
   :members:
   :exclude-members: model_config, model_extra, model_fields
   :undoc-members:
   :inherited-members:
   :show-inheritance:
   
----


**ModuleConfig**
================

.. autoclass:: prescyent.predictor.lightning.configs.module_config.ModuleConfig
   :members:
   :exclude-members: model_config, model_extra, model_fields
   :undoc-members:
   :inherited-members:
   :show-inheritance:


----

**Seq2SeqConfig**
=================

.. autoclass:: prescyent.predictor.lightning.models.sequence.seq2seq.config.Seq2SeqConfig
   :members:
   :exclude-members: model_config, model_extra, model_fields
   :undoc-members:
   :inherited-members:
   :show-inheritance:

----

**SiMLPeConfig**
================

.. autoclass:: prescyent.predictor.lightning.models.sequence.simlpe.config.SiMLPeConfig
   :members:
   :exclude-members: model_config, model_extra, model_fields
   :undoc-members:
   :inherited-members:
   :show-inheritance:

----

**MlpConfig**
=============

.. autoclass:: prescyent.predictor.lightning.models.sequence.mlp.config.MlpConfig
   :members:
   :exclude-members: model_config, model_extra, model_fields
   :undoc-members:
   :inherited-members:
   :show-inheritance:

----

**SARLSTMConfig**
=================

.. autoclass:: prescyent.predictor.lightning.models.autoreg.sarlstm.config.SARLSTMConfig
   :members:
   :exclude-members: model_config, model_extra, model_fields, name_sup_or_equal_one
   :undoc-members:
   :inherited-members:
   :show-inheritance:

********
Training
********


**OptimizerConfig**
===================

.. autoclass:: prescyent.predictor.lightning.configs.optimizer_config.OptimizerConfig
   :members:
   :exclude-members: model_config, model_extra, model_fields
   :undoc-members:
   :inherited-members:
   :show-inheritance:

----

**TrainingConfig**
==================

.. autoclass:: prescyent.predictor.lightning.configs.training_config.TrainingConfig
   :members:
   :exclude-members: model_config, model_extra, model_fields
   :undoc-members:
   :inherited-members:
   :show-inheritance: