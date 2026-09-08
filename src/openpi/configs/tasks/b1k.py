from . import TASK_REGISTRY

# i3l task prompts
TASKS = {
    "turning_on_radio": "Turn on the radio receiver that's on the table in the living room.",
    "putting_dishes_away_after_cleaning": (
        "In the kitchen, gather all eight plates from the two countertops, place them all inside a single cabinet "
        "(either one), and make sure all cabinets are closed when you're done."
    ),
    "carrying_in_groceries": (
        "Take the sack of groceries out of the car trunk in the garage, bring it to the kitchen, and put both the "
        "tomato and the carton of milk into the refrigerator in the kitchen. When you're done, close the car trunk "
        "and make sure the refrigerator in the kitchen is closed."
    ),
    "bringing_water": (
        "Retrieve the two bottles from the refrigerator in the kitchen, bring them to the living room, and place "
        "both on the coffee table. Make sure the refrigerator is closed when you finish."
    ),
}

# Register in global registry
TASK_REGISTRY["b1k"] = TASKS
