import importlib.util
import sys
from pathlib import Path


def load_runtime():
    root = Path(__file__).resolve().parents[2]
    upstream = root/'external/OneIE'
    sys.path.insert(0, str(upstream))
    # Use a collision-free label radix. Global features are disabled for this run.
    for name in ('util', 'model'):
        source = (upstream/f'{name}.py').read_text(encoding='utf-8')
        source = source.replace('* 100 +', '* 100000 +')
        if name == 'util':
            for var in ('entity_type_set', 'event_type_set', 'relation_type_set', 'role_type_set'):
                source = source.replace(f'enumerate({var}, 1)', f'enumerate(sorted({var}), 1)')
                source = source.replace(f'for t in {var}:', f'for t in sorted({var}):')
        if name == 'model':
            source = source.replace('generate_global_feature_maps(vocabs, valid_patterns)',
                                    "generate_global_feature_maps(vocabs, valid_patterns) if self.use_global_features else {}")
        spec = importlib.util.spec_from_loader(name, loader=None)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        exec(compile(source, str(upstream/f'{name}.py'), 'exec'), module.__dict__)
    from data import IEDataset
    from config import Config
    from transformers import BertConfig
    Config.bert_config = property(lambda self: BertConfig.from_pretrained(root / "data/oneie/backbone"))
    from model import OneIE
    from util import generate_vocabs, load_valid_patterns
    return IEDataset, Config, OneIE, generate_vocabs, load_valid_patterns
