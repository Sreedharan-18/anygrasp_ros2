class GraspQualityConfigFactory:
    @staticmethod
    def create_config(*a, **k):
        class _Cfg: pass
        return _Cfg()
