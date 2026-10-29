
# Import and initialize the algorithms implemented in this benchmark.


def call_algo(algo_name, config, mode, device):

    algo_name = algo_name.lower()

    # ============================================================
    # MODE 0: ONLINE-TO-ONLINE REINFORCEMENT LEARNING
    # ============================================================
    if mode == 0:

        assert algo_name in [
            'sac',
            'sac_target_only',
            'darc',
            'vgdf',
            'sac_iw',
            'par',
        ], f"Unsupported mode 0 algorithm: {algo_name}"

        from online_online.darc import DARC
        from online_online.sac import SAC
        from online_online.sac_target_only import SACTargetOnly
        from online_online.vgdf import VGDF
        from online_online.sac_iw import SAC_IW
        from online_online.par import PAR

        algo_to_call = {
            'sac': SAC,
            'sac_target_only': SACTargetOnly,
            'darc': DARC,
            'vgdf': VGDF,
            'sac_iw': SAC_IW,
            'par': PAR,
        }

    
    # ============================================================
    # MODE 1: OFFLINE-TO-ONLINE REINFORCEMENT LEARNING
    # ============================================================
    elif mode == 1:

        # Each algorithm is imported only when selected.
        # This prevents missing optional modules from breaking
        # unrelated algorithms such as EA-NTC.

        algorithm_modules = {
            'cql_sac': (
                'offline_online.cql_sac', 'CQLSAC'
            ),
            'bc_vgdf': (
                'offline_online.bc_vgdf', 'BCVGDF'
            ),
            'bc_sac': (
                'offline_online.bc_sac', 'BCSAC'
            ),
            'ea_ntc_bc_sac': (
                'offline_online.ea_ntc_bc_sac',
                'EANTCBCSAC'
            ),
            'ucdgt_bc_sac': (
                'offline_online.ucdgt_bc_sac',
                'UCDGTBCSAC'
            ),
            'ucdgt_adaptive_tau': (
                'offline_online.ucdgt_adaptive_tau',
                'UCDGTAdaptiveTau'
            ),
            'h2o': (
                'offline_online.h2o', 'H2O'
            ),
            'mcq_sac': (
                'offline_online.mcq_sac', 'MCQSAC'
            ),
            'rlpd': (
                'offline_online.rlpd', 'RLPD'
            ),
            'bc_par': (
                'offline_online.bc_par', 'BCPAR'
            ),
            'bc_par_no_penalty': (
                'offline_online.bc_par_no_penalty',
                'BCPARNoPenalty'
            ),
        }

        if algo_name not in algorithm_modules:
            raise ValueError(
                f"Unsupported mode 1 algorithm: {algo_name}"
            )

        import importlib

        module_name, class_name = algorithm_modules[algo_name]

        module = importlib.import_module(module_name)
        algorithm_class = getattr(module, class_name)

        return algorithm_class(config, device)


    # ============================================================
    # MODE 2: ONLINE-TO-OFFLINE REINFORCEMENT LEARNING
    # ============================================================
    elif mode == 2:

        assert algo_name in [
            'sac_bc',
            'h2o',
            'sac_cql',
            'sac_mcq',
            'par_bc',
        ], f"Unsupported mode 2 algorithm: {algo_name}"

        from online_offline.sac_bc import SACBC
        from online_offline.h2o import H2O
        from online_offline.sac_cql import SACCQL
        from online_offline.sac_mcq import SACMCQ
        from online_offline.par_bc import PARBC

        algo_to_call = {
            'sac_bc': SACBC,
            'h2o': H2O,
            'sac_cql': SACCQL,
            'sac_mcq': SACMCQ,
            'par_bc': PARBC,
        }

    # ============================================================
    # MODE 3: OFFLINE-TO-OFFLINE REINFORCEMENT LEARNING
    # ============================================================
    elif mode == 3:

        assert algo_name in [
            'dara',
            'bosa',
            'iql',
            'td3_bc',
            'igdf',
        ], f"Unsupported mode 3 algorithm: {algo_name}"

        from offline_offline.dara import DARA
        from offline_offline.bosa import BOSA
        from offline_offline.iql import IQL
        from offline_offline.td3_bc import TD3BC
        from offline_offline.igdf import IGDF

        algo_to_call = {
            'dara': DARA,
            'bosa': BOSA,
            'iql': IQL,
            'td3_bc': TD3BC,
            'igdf': IGDF,
        }

    else:
        raise ValueError(
            f"Unsupported training mode: {mode}. "
            "Valid modes are 0, 1, 2, and 3."
        )

    # ============================================================
    # INSTANTIATE THE SELECTED ALGORITHM
    # ============================================================
    algorithm_class = algo_to_call[algo_name]

    policy = algorithm_class(config, device)

    return policy

