from src.attacks.bandwagon import BandwagonAttack
from src.attacks.cfair_favorite import CFairFavoriteAttack
from src.attacks.cfair_influencer import CFairInfluencerAttack
from src.attacks.cfair_reverse_favorite import CFairReverseFavoriteAttack
from src.attacks.cfair_sampling import CFairSamplingAttack
from src.attacks.power_user import PowerUserAttack
from src.attacks.reverse_bandwagon import ReverseBandwagonAttack


CFAIR_ATTACK_REGISTRY = {
    "CFairInfluencerAttack": CFairInfluencerAttack,
    "CFairFavoriteAttack": CFairFavoriteAttack,
    "CFairReverseFavoriteAttack": CFairReverseFavoriteAttack,
    "CFairSamplingAttack": CFairSamplingAttack,
}

# Vanilla attacks: standard shilling heuristics
VANILLA_ATTACK_REGISTRY = {
    "PowerUserAttack": PowerUserAttack,
    "BandwagonAttack": BandwagonAttack,
    "ReverseBandwagonAttack": ReverseBandwagonAttack,
}
