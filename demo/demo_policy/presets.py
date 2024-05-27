presets = [
    {
        "type": "text",
        "value": "Use active verbs.",
        "examples": [
            {"type": "positive", "value": "I logged into the account."},
            {"type": "negative", "value": "The account was logged into by me."},
        ],
    },
    {
        "type": "text",
        "value": "Keep sentences under 20 words.",
        "examples": [
            {
                "type": "positive",
                "value": "At sunset, the tranquil lake glowed warmly while a gentle breeze carried the scent of pine and wildflowers through the trees.",
            },
            {
                "type": "negative",
                "value": "As the sun dipped below the horizon, casting a warm glow over the tranquil lake, a gentle breeze whispered through the trees, carrying the scent of pine and wildflowers.",
            },
        ],
    },
    {
        "type": "text",
        "value": "Write in American English.",
        "examples": [
            {"type": "positive", "value": "Organization, program, and color"},
            {"type": "negative", "value": "Organisation, programme, and colour"},
        ],
    },
    {
        "type": "text",
        "value": "Do not capitalize random words in the middle of sentences, unless that is grammatically correct.",
        "examples": [
            {
                "type": "positive",
                "value": "In my opinion, I believe that the meeting proceeded quite smoothly.",
            },
            {
                "type": "negative",
                "value": "In my opinion, I believe that the Meeting proceeded quite smoothly.",
            },
        ],
    },
    {
        "type": "text",
        "value": "Spell out numbers when they begin a sentence; otherwise, use numerals.",
        "examples": [
            {"type": "positive", "value": "Ten new employees started on Monday."},
            {"type": "negative", "value": "10 new employees started on Monday."},
        ],
    },
    {
        "type": "text",
        "value": "Avoid idioms relating to disability, age, gender, ethnicity, or religion.",
        "examples": [
            {
                "type": "positive",
                "value": "The seller provided excellent customer service.",
            },
            {
                "type": "negative",
                "value": "The salesman provided excellent customer service.",
            },
        ],
    },
    {
        "type": "text",
        "value": "Spell out abbreviations upon first mention. Beyond that, use the abbreviation.",
        "examples": [
            {
                "type": "positive",
                "value": "Let’s meet at 10 o’clock Coordinated Universal Time (UTC).",
            },
            {"type": "negative", "value": "Let’s meet at 10 o’clock UTC."},
        ],
    },
]

test_cases = [
    {
        "text": "The novel was authored by the renowned writer, showcasing his mastery of storytelling techniques.",
        "explanation": [
            {
                "violates": "Use active verbs",
                "fix": "Restructure the sentence to use an active verb, e.g., 'The renowned writer authored the novel, showcasing his mastery of storytelling techniques.'",
            }
        ],
    },
    {
        "text": "As the clock struck midnight, signaling the end of another day, a lone figure emerged from the shadows, shrouded in mystery and intrigue.",
        "explanation": [
            {
                "violates": "Keep sentences under 20 words",
                "fix": "Break the sentence into two or more shorter sentences to maintain clarity and readability.",
            }
        ],
    },
    {
        "text": "Theatre, favourite, and analyse",
        "explanation": [
            {
                "violates": "Write in American English",
                "fix": "Replace with American English equivalents: 'Theater, favorite, and analyze.'",
            }
        ],
    },
    {
        "text": "In my humble opinion, I believe that the Proposal is worth considering.",
        "explanation": [
            {
                "violates": "Do not capitalize random words in the middle of sentences",
                "fix": "Avoid unnecessary capitalization: 'In my humble opinion, I believe that the proposal is worth considering.'",
            }
        ],
    },
    {
        "text": "27 participants attended the conference, eager to exchange ideas and insights.",
        "explanation": [
            {
                "violates": "Spell out numbers when they begin a sentence; otherwise, use numerals",
                "fix": "Spell out the number at the beginning of the sentence: 'Twenty-seven participants attended the conference, eager to exchange ideas and insights.'",
            }
        ],
    },
    {
        "text": "The fireman bravely rescued the family from the burning building.",
        "explanation": [
            {
                "violates": "Avoid idioms relating to disability, age, gender, ethnicity, or religion",
                "fix": "Use neutral language: 'The firefighter bravely rescued the family from the burning building.'",
            }
        ],
    },
    {
        "text": "The NASA administrator unveiled plans for a manned mission to Mars during the press conference.",
        "explanation": [
            {
                "violates": "Spell out abbreviations upon first mention",
                "fix": "Spell out the abbreviation upon first mention: 'The National Aeronautics and Space Administration (NASA) administrator unveiled plans for a manned mission to Mars during the press conference.'",
            }
        ],
    },
]
