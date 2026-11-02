"""Generated rational XC compositions; run tools/generate_semilocal_compositions.py."""

from typing import TypedDict


class CompositionRecord(TypedDict):
    """Serialized recipe and its deterministic native registration."""

    components: tuple[tuple[str, str], ...]
    exact_exchange: str
    code: int
    methods: list[str]


# fmt: off
COMPOSITIONS: dict[str, CompositionRecord] = {'COMPOSED_15A13F7133C6F808': {'code': 1096422391,
                               'components': (('MGGA_C_RSCAN', '1'),
                                              ('MGGA_X_RSCAN', '1')),
                               'exact_exchange': '0',
                               'methods': ['RSCAN']},
 'COMPOSED_1FE8C28BA34EEA44': {'code': 1107201064,
                               'components': (('MGGA_C_R2SCAN', '1'),
                                              ('MGGA_X_R2SCAN', '1')),
                               'exact_exchange': '0',
                               'methods': ['R2SCAN']},
 'COMPOSED_355ADABFCCE8A66E': {'code': 1129688491,
                               'components': (('GGA_C_PBE', '1'), ('GGA_X_PBE', '1/2')),
                               'exact_exchange': '1/2',
                               'methods': ['PBE50']},
 'COMPOSED_58FA5968A2DF3536': {'code': 1167041942,
                               'components': (('GGA_C_LYP', '1'), ('GGA_X_B88', '1')),
                               'exact_exchange': '0',
                               'methods': ['BLYP']},
 'COMPOSED_65220253200503B0': {'code': 1179787301,
                               'components': (('MGGA_C_SCAN', '1'),
                                              ('MGGA_X_SCAN', '1')),
                               'exact_exchange': '0',
                               'methods': ['SCAN']},
 'COMPOSED_6DC6F99022347467': {'code': 1188851609,
                               'components': (('GGA_C_PBE', '1'), ('GGA_X_PBE', '3/4')),
                               'exact_exchange': '1/4',
                               'methods': ['PBE0']},
 'COMPOSED_76E67D3109F41A25': {'code': 1198417875,
                               'components': (('GGA_C_PW91', '1'), ('GGA_X_PW91', '1')),
                               'exact_exchange': '0',
                               'methods': ['PW91']},
 'COMPOSED_79DD37AA6B9C7ECA': {'code': 1201525626,
                               'components': (('MGGA_C_RSCAN', '1'),
                                              ('MGGA_X_RSCAN', '3/4')),
                               'exact_exchange': '1/4',
                               'methods': ['RSCAN0']},
 'COMPOSED_7F0AC67EF56CB630': {'code': 1206955111,
                               'components': (('GGA_C_LYP', '1'), ('GGA_X_B88', '1/2')),
                               'exact_exchange': '1/2',
                               'methods': ['BHANDHLYP']},
 'COMPOSED_98507B0892967D9E': {'code': 1233455024,
                               'components': (('GGA_C_PBE', '1'), ('GGA_X_PBE', '1')),
                               'exact_exchange': '0',
                               'methods': ['PBE']},
 'COMPOSED_9928C773C6393988': {'code': 1234340983,
                               'components': (('MGGA_C_R2SCAN', '1'),
                                              ('MGGA_X_R2SCAN', '3/4')),
                               'exact_exchange': '1/4',
                               'methods': ['R2SCAN0']},
 'COMPOSED_A96AA0CA52D179F3': {'code': 1251387916,
                               'components': (('GGA_C_P86', '1'), ('GGA_X_B88', '1')),
                               'exact_exchange': '0',
                               'methods': ['BP86']},
 'COMPOSED_B3D919511EDD79B2': {'code': 1262326165,
                               'components': (('MGGA_C_R2SCAN', '1'),
                                              ('MGGA_X_R2SCAN', '1/2')),
                               'exact_exchange': '1/2',
                               'methods': ['R2SCAN50']},
 'COMPOSED_C0ED816D77E25CCB': {'code': 1276041238,
                               'components': (('MGGA_C_SCAN', '1'),
                                              ('MGGA_X_SCAN', '3/4')),
                               'exact_exchange': '1/4',
                               'methods': ['SCAN0']},
 'COMPOSED_E03B732D8FF8E385': {'code': 1308866354,
                               'components': (('MGGA_C_R2SCAN', '1'),
                                              ('MGGA_X_R2SCAN', '9/10')),
                               'exact_exchange': '1/10',
                               'methods': ['R2SCANH']}}
