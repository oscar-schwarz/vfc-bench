from vfc_bench.data.function_sets.smart_home import SmartHome
from vfc_bench.dataset import TestCase

CASE = TestCase(
    transcription="ugh someone's gotta setup the heater in my bedroom before I'm home. It's freezing",
    setup=lambda: SmartHome(rooms=["living room", "bedroom", "kitchen", "bathroom"]),
    expect={
        "heater is on": lambda s: s.rooms["bedroom"].heater.on,
        "bedroom is warm": lambda s: s.rooms["bedroom"].heater.target_c >= 21,
    },
)
