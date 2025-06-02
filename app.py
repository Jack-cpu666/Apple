from flask import Flask, render_template_string, request, session, redirect, url_for
import random

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'

# Add enumerate to Jinja2 environment
app.jinja_env.globals.update(enumerate=enumerate)

# All 100 civics questions and answers from the official USCIS study guide
CIVICS_QUESTIONS = {
    1: {
        "question": "What is the supreme law of the land?",
        "answers": ["the Constitution"],
        "correct": "the Constitution"
    },
    2: {
        "question": "What does the Constitution do?",
        "answers": ["sets up the government", "defines the government", "protects basic rights of Americans"],
        "correct": "sets up the government"
    },
    3: {
        "question": "The idea of self-government is in the first three words of the Constitution. What are these words?",
        "answers": ["We the People"],
        "correct": "We the People"
    },
    4: {
        "question": "What is an amendment?",
        "answers": ["a change (to the Constitution)", "an addition (to the Constitution)"],
        "correct": "a change (to the Constitution)"
    },
    5: {
        "question": "What do we call the first ten amendments to the Constitution?",
        "answers": ["the Bill of Rights"],
        "correct": "the Bill of Rights"
    },
    6: {
        "question": "What is one right or freedom from the First Amendment?",
        "answers": ["speech", "religion", "assembly", "press", "petition the government"],
        "correct": "speech"
    },
    7: {
        "question": "How many amendments does the Constitution have?",
        "answers": ["twenty-seven (27)"],
        "correct": "twenty-seven"
    },
    8: {
        "question": "What did the Declaration of Independence do?",
        "answers": ["announced our independence (from Great Britain)", "declared our independence (from Great Britain)", "said that the United States is free (from Great Britain)"],
        "correct": "announced our independence (from Great Britain)"
    },
    9: {
        "question": "What are two rights in the Declaration of Independence?",
        "answers": ["life", "liberty", "pursuit of happiness"],
        "correct": "life and liberty"
    },
    10: {
        "question": "What is freedom of religion?",
        "answers": ["You can practice any religion, or not practice a religion."],
        "correct": "You can practice any religion, or not practice a religion."
    },
    11: {
        "question": "What is the economic system in the United States?",
        "answers": ["capitalist economy", "market economy"],
        "correct": "capitalist economy"
    },
    12: {
        "question": "What is the \"rule of law\"?",
        "answers": ["Everyone must follow the law.", "Leaders must obey the law.", "Government must obey the law.", "No one is above the law."],
        "correct": "Everyone must follow the law."
    },
    13: {
        "question": "Name one branch or part of the government.",
        "answers": ["Congress", "legislative", "President", "executive", "the courts", "judicial"],
        "correct": "Congress"
    },
    14: {
        "question": "What stops one branch of government from becoming too powerful?",
        "answers": ["checks and balances", "separation of powers"],
        "correct": "checks and balances"
    },
    15: {
        "question": "Who is in charge of the executive branch?",
        "answers": ["the President"],
        "correct": "the President"
    },
    16: {
        "question": "Who makes federal laws?",
        "answers": ["Congress", "Senate and House (of Representatives)", "(U.S. or national) legislature"],
        "correct": "Congress"
    },
    17: {
        "question": "What are the two parts of the U.S. Congress?",
        "answers": ["the Senate and House (of Representatives)"],
        "correct": "the Senate and House (of Representatives)"
    },
    18: {
        "question": "How many U.S. Senators are there?",
        "answers": ["one hundred (100)"],
        "correct": "one hundred"
    },
    19: {
        "question": "We elect a U.S. Senator for how many years?",
        "answers": ["six (6)"],
        "correct": "six"
    },
    20: {
        "question": "Who is one of your state's U.S. Senators now?",
        "answers": ["Answers will vary"],
        "correct": "Answers will vary"
    },
    21: {
        "question": "The House of Representatives has how many voting members?",
        "answers": ["four hundred thirty-five (435)"],
        "correct": "four hundred thirty-five"
    },
    22: {
        "question": "We elect a U.S. Representative for how many years?",
        "answers": ["two (2)"],
        "correct": "two"
    },
    23: {
        "question": "Name your U.S. Representative.",
        "answers": ["Answers will vary"],
        "correct": "Answers will vary"
    },
    24: {
        "question": "Who does a U.S. Senator represent?",
        "answers": ["all people of the state"],
        "correct": "all people of the state"
    },
    25: {
        "question": "Why do some states have more Representatives than other states?",
        "answers": ["(because of) the state's population", "(because) they have more people", "(because) some states have more people"],
        "correct": "because of the state's population"
    },
    26: {
        "question": "We elect a President for how many years?",
        "answers": ["four (4)"],
        "correct": "four"
    },
    27: {
        "question": "In what month do we vote for President?",
        "answers": ["November"],
        "correct": "November"
    },
    28: {
        "question": "What is the name of the President of the United States now?",
        "answers": ["Donald Trump"],
        "correct": "Donald Trump"
    },
    29: {
        "question": "What is the name of the Vice President of the United States now?",
        "answers": ["J.D. Vance"],
        "correct": "J.D. Vance"
    },
    30: {
        "question": "If the President can no longer serve, who becomes President?",
        "answers": ["the Vice President"],
        "correct": "the Vice President"
    },
    31: {
        "question": "If both the President and the Vice President can no longer serve, who becomes President?",
        "answers": ["the Speaker of the House"],
        "correct": "the Speaker of the House"
    },
    32: {
        "question": "Who is the Commander in Chief of the military?",
        "answers": ["the President"],
        "correct": "the President"
    },
    33: {
        "question": "Who signs bills to become laws?",
        "answers": ["the President"],
        "correct": "the President"
    },
    34: {
        "question": "Who vetoes bills?",
        "answers": ["the President"],
        "correct": "the President"
    },
    35: {
        "question": "What does the President's Cabinet do?",
        "answers": ["advises the President"],
        "correct": "advises the President"
    },
    36: {
        "question": "What are two Cabinet-level positions?",
        "answers": ["Secretary of Agriculture", "Secretary of Commerce", "Secretary of Defense", "Secretary of Education", "Secretary of Energy", "Secretary of Health and Human Services", "Secretary of Homeland Security", "Secretary of Housing and Urban Development", "Secretary of the Interior", "Secretary of Labor", "Secretary of State", "Secretary of Transportation", "Secretary of the Treasury", "Secretary of Veterans Affairs", "Attorney General", "Vice President"],
        "correct": "Secretary of State and Secretary of Defense"
    },
    37: {
        "question": "What does the judicial branch do?",
        "answers": ["reviews laws", "explains laws", "resolves disputes (disagreements)", "decides if a law goes against the Constitution"],
        "correct": "reviews laws"
    },
    38: {
        "question": "What is the highest court in the United States?",
        "answers": ["the Supreme Court"],
        "correct": "the Supreme Court"
    },
    39: {
        "question": "How many justices are on the Supreme Court?",
        "answers": ["nine (9)"],
        "correct": "nine"
    },
    40: {
        "question": "Who is the Chief Justice of the United States now?",
        "answers": ["John Roberts"],
        "correct": "John Roberts"
    },
    41: {
        "question": "Under our Constitution, some powers belong to the federal government. What is one power of the federal government?",
        "answers": ["to print money", "to declare war", "to create an army", "to make treaties"],
        "correct": "to print money"
    },
    42: {
        "question": "Under our Constitution, some powers belong to the states. What is one power of the states?",
        "answers": ["provide schooling and education", "provide protection (police)", "provide safety (fire departments)", "give a driver's license", "approve zoning and land use"],
        "correct": "provide schooling and education"
    },
    43: {
        "question": "Who is the Governor of your state now?",
        "answers": ["Answers will vary"],
        "correct": "Answers will vary"
    },
    44: {
        "question": "What is the capital of your state?",
        "answers": ["Answers will vary"],
        "correct": "Answers will vary"
    },
    45: {
        "question": "What are the two major political parties in the United States?",
        "answers": ["Democratic and Republican"],
        "correct": "Democratic and Republican"
    },
    46: {
        "question": "What is the political party of the President now?",
        "answers": ["Republican"],
        "correct": "Republican"
    },
    47: {
        "question": "What is the name of the Speaker of the House of Representatives now?",
        "answers": ["Mike Johnson"],
        "correct": "Mike Johnson"
    },
    48: {
        "question": "There are four amendments to the Constitution about who can vote. Describe one of them.",
        "answers": ["Citizens eighteen (18) and older (can vote).", "You don't have to pay (a poll tax) to vote.", "Any citizen can vote. (Women and men can vote.)", "A male citizen of any race (can vote)."],
        "correct": "Citizens eighteen (18) and older (can vote)."
    },
    49: {
        "question": "What is one responsibility that is only for United States citizens?",
        "answers": ["serve on a jury", "vote in a federal election"],
        "correct": "serve on a jury"
    },
    50: {
        "question": "Name one right only for United States citizens.",
        "answers": ["vote in a federal election", "run for federal office"],
        "correct": "vote in a federal election"
    },
    51: {
        "question": "What are two rights of everyone living in the United States?",
        "answers": ["freedom of expression", "freedom of speech", "freedom of assembly", "freedom to petition the government", "freedom of religion", "the right to bear arms"],
        "correct": "freedom of speech and freedom of religion"
    },
    52: {
        "question": "What do we show loyalty to when we say the Pledge of Allegiance?",
        "answers": ["the United States", "the flag"],
        "correct": "the United States"
    },
    53: {
        "question": "What is one promise you make when you become a United States citizen?",
        "answers": ["give up loyalty to other countries", "defend the Constitution and laws of the United States", "obey the laws of the United States", "serve in the U.S. military (if needed)", "serve (do important work for) the nation (if needed)", "be loyal to the United States"],
        "correct": "give up loyalty to other countries"
    },
    54: {
        "question": "How old do citizens have to be to vote for President?",
        "answers": ["eighteen (18) and older"],
        "correct": "eighteen"
    },
    55: {
        "question": "What are two ways that Americans can participate in their democracy?",
        "answers": ["vote", "join a political party", "help with a campaign", "join a civic group", "join a community group", "give an elected official your opinion on an issue", "call Senators and Representatives", "publicly support or oppose an issue or policy", "run for office", "write to a newspaper"],
        "correct": "vote and join a political party"
    },
    56: {
        "question": "When is the last day you can send in federal income tax forms?",
        "answers": ["April 15"],
        "correct": "April 15"
    },
    57: {
        "question": "When must all men register for the Selective Service?",
        "answers": ["at age eighteen (18)", "between eighteen (18) and twenty-six (26)"],
        "correct": "at age eighteen"
    },
    58: {
        "question": "What is one reason colonists came to America?",
        "answers": ["freedom", "political liberty", "religious freedom", "economic opportunity", "practice their religion", "escape persecution"],
        "correct": "freedom"
    },
    59: {
        "question": "Who lived in America before the Europeans arrived?",
        "answers": ["American Indians", "Native Americans"],
        "correct": "American Indians"
    },
    60: {
        "question": "What group of people was taken to America and sold as slaves?",
        "answers": ["Africans", "people from Africa"],
        "correct": "Africans"
    },
    61: {
        "question": "Why did the colonists fight the British?",
        "answers": ["because of high taxes (taxation without representation)", "because the British army stayed in their houses (boarding, quartering)", "because they didn't have self-government"],
        "correct": "because of high taxes"
    },
    62: {
        "question": "Who wrote the Declaration of Independence?",
        "answers": ["(Thomas) Jefferson"],
        "correct": "Thomas Jefferson"
    },
    63: {
        "question": "When was the Declaration of Independence adopted?",
        "answers": ["July 4, 1776"],
        "correct": "July 4, 1776"
    },
    64: {
        "question": "There were 13 original states. Name three.",
        "answers": ["New Hampshire", "Massachusetts", "Rhode Island", "Connecticut", "New York", "New Jersey", "Pennsylvania", "Delaware", "Maryland", "Virginia", "North Carolina", "South Carolina", "Georgia"],
        "correct": "New York, New Jersey, and Pennsylvania"
    },
    65: {
        "question": "What happened at the Constitutional Convention?",
        "answers": ["The Constitution was written.", "The Founding Fathers wrote the Constitution."],
        "correct": "The Constitution was written."
    },
    66: {
        "question": "When was the Constitution written?",
        "answers": ["1787"],
        "correct": "1787"
    },
    67: {
        "question": "The Federalist Papers supported the passage of the U.S. Constitution. Name one of the writers.",
        "answers": ["(James) Madison", "(Alexander) Hamilton", "(John) Jay", "Publius"],
        "correct": "James Madison"
    },
    68: {
        "question": "What is one thing Benjamin Franklin is famous for?",
        "answers": ["U.S. diplomat", "oldest member of the Constitutional Convention", "first Postmaster General of the United States", "writer of \"Poor Richard's Almanac\"", "started the first free libraries"],
        "correct": "U.S. diplomat"
    },
    69: {
        "question": "Who is the \"Father of Our Country\"?",
        "answers": ["(George) Washington"],
        "correct": "George Washington"
    },
    70: {
        "question": "Who was the first President?",
        "answers": ["(George) Washington"],
        "correct": "George Washington"
    },
    71: {
        "question": "What territory did the United States buy from France in 1803?",
        "answers": ["the Louisiana Territory", "Louisiana"],
        "correct": "the Louisiana Territory"
    },
    72: {
        "question": "Name one war fought by the United States in the 1800s.",
        "answers": ["War of 1812", "Mexican-American War", "Civil War", "Spanish-American War"],
        "correct": "Civil War"
    },
    73: {
        "question": "Name the U.S. war between the North and the South.",
        "answers": ["the Civil War", "the War between the States"],
        "correct": "the Civil War"
    },
    74: {
        "question": "Name one problem that led to the Civil War.",
        "answers": ["slavery", "economic reasons", "states' rights"],
        "correct": "slavery"
    },
    75: {
        "question": "What was one important thing that Abraham Lincoln did?",
        "answers": ["freed the slaves (Emancipation Proclamation)", "saved (or preserved) the Union", "led the United States during the Civil War"],
        "correct": "freed the slaves"
    },
    76: {
        "question": "What did the Emancipation Proclamation do?",
        "answers": ["freed the slaves", "freed slaves in the Confederacy", "freed slaves in the Confederate states", "freed slaves in most Southern states"],
        "correct": "freed the slaves"
    },
    77: {
        "question": "What did Susan B. Anthony do?",
        "answers": ["fought for women's rights", "fought for civil rights"],
        "correct": "fought for women's rights"
    },
    78: {
        "question": "Name one war fought by the United States in the 1900s.",
        "answers": ["World War I", "World War II", "Korean War", "Vietnam War", "(Persian) Gulf War"],
        "correct": "World War II"
    },
    79: {
        "question": "Who was President during World War I?",
        "answers": ["(Woodrow) Wilson"],
        "correct": "Woodrow Wilson"
    },
    80: {
        "question": "Who was President during the Great Depression and World War II?",
        "answers": ["(Franklin) Roosevelt"],
        "correct": "Franklin Roosevelt"
    },
    81: {
        "question": "Who did the United States fight in World War II?",
        "answers": ["Japan, Germany, and Italy"],
        "correct": "Japan, Germany, and Italy"
    },
    82: {
        "question": "Before he was President, Eisenhower was a general. What war was he in?",
        "answers": ["World War II"],
        "correct": "World War II"
    },
    83: {
        "question": "During the Cold War, what was the main concern of the United States?",
        "answers": ["Communism"],
        "correct": "Communism"
    },
    84: {
        "question": "What movement tried to end racial discrimination?",
        "answers": ["civil rights (movement)"],
        "correct": "civil rights movement"
    },
    85: {
        "question": "What did Martin Luther King, Jr. do?",
        "answers": ["fought for civil rights", "worked for equality for all Americans"],
        "correct": "fought for civil rights"
    },
    86: {
        "question": "What major event happened on September 11, 2001, in the United States?",
        "answers": ["Terrorists attacked the United States."],
        "correct": "Terrorists attacked the United States."
    },
    87: {
        "question": "Name one American Indian tribe in the United States.",
        "answers": ["Cherokee", "Navajo", "Sioux", "Chippewa", "Choctaw", "Pueblo", "Apache", "Iroquois", "Creek", "Blackfeet", "Seminole", "Cheyenne", "Arawak", "Shawnee", "Mohegan", "Huron", "Oneida", "Lakota", "Crow", "Teton", "Hopi", "Inuit"],
        "correct": "Cherokee"
    },
    88: {
        "question": "Name one of the two longest rivers in the United States.",
        "answers": ["Missouri (River)", "Mississippi (River)"],
        "correct": "Mississippi River"
    },
    89: {
        "question": "What ocean is on the West Coast of the United States?",
        "answers": ["Pacific (Ocean)"],
        "correct": "Pacific Ocean"
    },
    90: {
        "question": "What ocean is on the East Coast of the United States?",
        "answers": ["Atlantic (Ocean)"],
        "correct": "Atlantic Ocean"
    },
    91: {
        "question": "Name one U.S. territory.",
        "answers": ["Puerto Rico", "U.S. Virgin Islands", "American Samoa", "Northern Mariana Islands", "Guam"],
        "correct": "Puerto Rico"
    },
    92: {
        "question": "Name one state that borders Canada.",
        "answers": ["Maine", "New Hampshire", "Vermont", "New York", "Pennsylvania", "Ohio", "Michigan", "Minnesota", "North Dakota", "Montana", "Idaho", "Washington", "Alaska"],
        "correct": "New York"
    },
    93: {
        "question": "Name one state that borders Mexico.",
        "answers": ["California", "Arizona", "New Mexico", "Texas"],
        "correct": "California"
    },
    94: {
        "question": "What is the capital of the United States?",
        "answers": ["Washington, D.C."],
        "correct": "Washington, D.C."
    },
    95: {
        "question": "Where is the Statue of Liberty?",
        "answers": ["New York (Harbor)", "Liberty Island"],
        "correct": "New York Harbor"
    },
    96: {
        "question": "Why does the flag have 13 stripes?",
        "answers": ["because there were 13 original colonies", "because the stripes represent the original colonies"],
        "correct": "because there were 13 original colonies"
    },
    97: {
        "question": "Why does the flag have 50 stars?",
        "answers": ["because there is one star for each state", "because each star represents a state", "because there are 50 states"],
        "correct": "because there are 50 states"
    },
    98: {
        "question": "What is the name of the national anthem?",
        "answers": ["The Star-Spangled Banner"],
        "correct": "The Star-Spangled Banner"
    },
    99: {
        "question": "When do we celebrate Independence Day?",
        "answers": ["July 4"],
        "correct": "July 4"
    },
    100: {
        "question": "Name two national U.S. holidays.",
        "answers": ["New Year's Day", "Martin Luther King, Jr. Day", "Presidents' Day", "Memorial Day", "Independence Day", "Labor Day", "Columbus Day", "Veterans Day", "Thanksgiving", "Christmas"],
        "correct": "Independence Day and Thanksgiving"
    }
}

# Reading vocabulary sentences
READING_SENTENCES = [
    "America is the land of freedom.",
    "Citizens have the right to vote.",
    "George Washington was the first President.",
    "The American flag has stars and stripes.",
    "Congress makes the laws.",
    "We celebrate Independence Day on July 4.",
    "The President lives in the White House.",
    "Abraham Lincoln was a great President.",
    "The Bill of Rights protects our freedoms.",
    "The United States has fifty states."
]

# Writing vocabulary sentences
WRITING_SENTENCES = [
    "America is a free country.",
    "Citizens can vote for President.",
    "George Washington was the Father of Our Country.",
    "The flag is red, white, and blue.",
    "Congress meets in Washington, D.C.",
    "We celebrate freedom on Independence Day.",
    "Lincoln freed the slaves.",
    "The President lives in the White House.",
    "Citizens have rights and freedoms.",
    "The United States has fifty states and Washington, D.C."
]

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>U.S. Naturalization Test Practice</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            position: relative;
            overflow-x: hidden;
        }
        
        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="50" cy="50" r="1" fill="rgba(255,255,255,0.1)"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
            pointer-events: none;
            z-index: -1;
        }
        
        .stars {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: -1;
        }
        
        .star {
            position: absolute;
            color: rgba(255, 255, 255, 0.8);
            animation: twinkle 3s infinite;
        }
        
        @keyframes twinkle {
            0%, 100% { opacity: 0.3; transform: scale(1); }
            50% { opacity: 1; transform: scale(1.2); }
        }
        
        .container {
            max-width: 1000px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 24px;
            box-shadow: 0 25px 60px rgba(0,0,0,0.3), 0 0 0 1px rgba(255,255,255,0.2);
            overflow: hidden;
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .header {
            background: linear-gradient(135deg, #1e3c72 0%, #c41e3a 50%, #003366 100%);
            color: white;
            padding: 40px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        
        .header::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            animation: shimmer 6s infinite;
        }
        
        @keyframes shimmer {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .header h1 {
            font-size: 3em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.5);
            font-weight: 700;
            position: relative;
            z-index: 1;
        }
        
        .header p {
            font-size: 1.3em;
            opacity: 0.95;
            font-weight: 500;
            position: relative;
            z-index: 1;
        }
        
        .content {
            padding: 50px;
        }
        
        .test-section {
            margin-bottom: 40px;
            padding: 40px;
            border: none;
            border-radius: 20px;
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            position: relative;
        }
        
        .test-section::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #c41e3a, #003366, #c41e3a);
            border-radius: 20px 20px 0 0;
        }
        
        .test-section h2 {
            color: #1e293b;
            margin-bottom: 30px;
            font-size: 2.2em;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .question-card {
            background: white;
            padding: 40px;
            margin: 30px 0;
            border-radius: 16px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            border: 1px solid rgba(0,0,0,0.05);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .question-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 6px;
            background: linear-gradient(90deg, #c41e3a, #003366);
        }
        
        .question-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.15);
        }
        
        .question-number {
            display: inline-block;
            background: linear-gradient(135deg, #c41e3a, #e74c3c);
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.9em;
            margin-bottom: 20px;
        }
        
        .question-text {
            font-size: 1.4em;
            color: #1e293b;
            margin-bottom: 30px;
            line-height: 1.6;
            font-weight: 500;
        }
        
        .answers-grid {
            display: grid;
            gap: 15px;
            margin-top: 25px;
        }
        
        .answer-option {
            padding: 20px 25px;
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 1.1em;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 15px;
            position: relative;
            overflow: hidden;
        }
        
        .answer-option::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(196, 30, 58, 0.1), transparent);
            transition: left 0.5s ease;
        }
        
        .answer-option:hover {
            background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
            border-color: #c41e3a;
            transform: translateX(5px);
            box-shadow: 0 5px 15px rgba(196, 30, 58, 0.2);
        }
        
        .answer-option:hover::before {
            left: 100%;
        }
        
        .answer-letter {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 35px;
            height: 35px;
            background: linear-gradient(135deg, #c41e3a, #e74c3c);
            color: white;
            border-radius: 50%;
            font-weight: 700;
            font-size: 1em;
        }
        
        .btn {
            background: linear-gradient(135deg, #c41e3a 0%, #e74c3c 100%);
            color: white;
            padding: 18px 35px;
            border: none;
            border-radius: 12px;
            font-size: 1.2em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin: 15px 10px;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 10px;
            box-shadow: 0 6px 20px rgba(196, 30, 58, 0.3);
        }
        
        .btn:hover {
            background: linear-gradient(135deg, #a91729 0%, #c0392b 100%);
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(196, 30, 58, 0.4);
        }
        
        .btn-secondary {
            background: linear-gradient(135deg, #003366 0%, #2a5298 100%);
            box-shadow: 0 6px 20px rgba(0, 51, 102, 0.3);
        }
        
        .btn-secondary:hover {
            background: linear-gradient(135deg, #002244 0%, #1e3c72 100%);
            box-shadow: 0 10px 30px rgba(0, 51, 102, 0.4);
        }
        
        .progress-container {
            margin: 30px 0;
            background: rgba(255,255,255,0.8);
            padding: 25px;
            border-radius: 16px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .progress-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .progress-text {
            font-weight: 600;
            color: #1e293b;
            font-size: 1.1em;
        }
        
        .progress-bar-container {
            background: #e2e8f0;
            border-radius: 25px;
            height: 12px;
            overflow: hidden;
            position: relative;
        }
        
        .progress-bar {
            background: linear-gradient(90deg, #c41e3a, #e74c3c);
            height: 100%;
            transition: width 0.6s ease;
            border-radius: 25px;
            position: relative;
        }
        
        .progress-bar::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
            animation: progressShine 2s infinite;
        }
        
        @keyframes progressShine {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        .feedback {
            margin-top: 30px;
            padding: 30px;
            border-radius: 16px;
            text-align: center;
            font-size: 1.2em;
            font-weight: 600;
            animation: slideIn 0.5s ease;
            position: relative;
            overflow: hidden;
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .feedback.correct {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            border: 2px solid #28a745;
            color: #155724;
        }
        
        .feedback.incorrect {
            background: linear-gradient(135deg, #f8d7da 0%, #f1c2c6 100%);
            border: 2px solid #dc3545;
            color: #721c24;
        }
        
        .feedback::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
            animation: feedbackShine 2s ease-in-out;
        }
        
        @keyframes feedbackShine {
            0% { left: -100%; }
            100% { left: 100%; }
        }
        
        .feedback-icon {
            font-size: 2em;
            margin-bottom: 15px;
            display: block;
        }
        
        .correct-answer {
            background: rgba(40, 167, 69, 0.1);
            border: 2px solid #28a745;
            color: #155724;
            padding: 20px;
            border-radius: 12px;
            margin-top: 20px;
            font-weight: 600;
        }
        
        .home-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px;
            margin-top: 40px;
        }
        
        .home-card {
            background: white;
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            text-align: center;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .home-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 6px;
            background: linear-gradient(90deg, #c41e3a, #003366);
        }
        
        .home-card:hover {
            transform: translateY(-10px);
            box-shadow: 0 20px 50px rgba(0,0,0,0.2);
        }
        
        .home-card-icon {
            font-size: 3em;
            margin-bottom: 20px;
            background: linear-gradient(135deg, #c41e3a, #003366);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .home-card h3 {
            font-size: 1.5em;
            margin-bottom: 15px;
            color: #1e293b;
            font-weight: 700;
        }
        
        .home-card p {
            color: #64748b;
            margin-bottom: 25px;
            line-height: 1.6;
        }
        
        .final-score {
            text-align: center;
            padding: 50px;
            background: white;
            border-radius: 20px;
            margin: 30px 0;
            box-shadow: 0 15px 40px rgba(0,0,0,0.1);
        }
        
        .score-circle {
            width: 150px;
            height: 150px;
            border-radius: 50%;
            margin: 0 auto 30px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2.5em;
            font-weight: 700;
            color: white;
        }
        
        .score-circle.pass {
            background: linear-gradient(135deg, #28a745, #20c997);
            box-shadow: 0 10px 30px rgba(40, 167, 69, 0.3);
        }
        
        .score-circle.fail {
            background: linear-gradient(135deg, #dc3545, #e74c3c);
            box-shadow: 0 10px 30px rgba(220, 53, 69, 0.3);
        }
        
        .flag-decoration {
            text-align: center;
            margin: 30px 0;
            font-size: 2.5em;
            animation: wave 2s ease-in-out infinite;
        }
        
        @keyframes wave {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }
        
        @media (max-width: 768px) {
            .content { padding: 25px; }
            .header h1 { font-size: 2em; }
            .question-text { font-size: 1.2em; }
            .home-cards { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="stars">
        <i class="fas fa-star star" style="top: 10%; left: 15%; animation-delay: 0s;"></i>
        <i class="fas fa-star star" style="top: 20%; left: 80%; animation-delay: 0.5s;"></i>
        <i class="fas fa-star star" style="top: 60%; left: 10%; animation-delay: 1s;"></i>
        <i class="fas fa-star star" style="top: 80%; left: 75%; animation-delay: 1.5s;"></i>
        <i class="fas fa-star star" style="top: 30%; left: 90%; animation-delay: 2s;"></i>
    </div>
    <div class="container">
        <div class="header">
            <h1>🇺🇸 U.S. Naturalization Test</h1>
            <p>Official Practice Test - USCIS Civics & English</p>
        </div>
        
        <div class="content">
                <div class="container">
        <div class="header">
            <h1><i class="fas fa-flag-usa"></i> U.S. Naturalization Test</h1>
            <p>Official Practice Test - USCIS Civics & English</p>
        </div>
        
        <div class="content">
            {% if page == 'home' %}
            <div class="test-section">
                <h2><i class="fas fa-home"></i> Welcome to Your Citizenship Journey</h2>
                <div style="background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); border: 2px solid #ffc107; color: #856404; padding: 25px; border-radius: 16px; margin-bottom: 30px;">
                    <h3 style="margin-bottom: 15px; color: #b8860b;"><i class="fas fa-info-circle"></i> Test Requirements:</h3>
                    <ul style="margin-left: 20px; margin-top: 10px; line-height: 1.8;">
                        <li><strong>Civics Test:</strong> Answer 6 out of 10 questions correctly</li>
                        <li><strong>English Reading:</strong> Read 1 out of 3 sentences correctly</li>
                        <li><strong>English Writing:</strong> Write 1 out of 3 sentences correctly</li>
                        <li><strong>Speaking:</strong> Demonstrated during the interview</li>
                    </ul>
                </div>
                
                <div class="flag-decoration">🗽⭐🦅⭐🗽</div>
                
                <div class="home-cards">
                    <div class="home-card">
                        <div class="home-card-icon">
                            <i class="fas fa-landmark"></i>
                        </div>
                        <h3>Civics Test</h3>
                        <p>Test your knowledge of U.S. history and government with official USCIS questions</p>
                        <a href="/civics" class="btn">
                            <i class="fas fa-play"></i> Start Civics Test
                        </a>
                    </div>
                    
                    <div class="home-card">
                        <div class="home-card-icon">
                            <i class="fas fa-book"></i>
                        </div>
                        <h3>English Test</h3>
                        <p>Practice reading and writing with official vocabulary from the naturalization test</p>
                        <a href="/english" class="btn btn-secondary">
                            <i class="fas fa-language"></i> Start English Test
                        </a>
                    </div>
                </div>
            </div>
            {% endif %}
            
            {% if page == 'civics_question' %}
            <div class="test-section">
                <h2><i class="fas fa-landmark"></i> Civics Test</h2>
                
                <div class="progress-container">
                    <div class="progress-header">
                        <span class="progress-text">Question {{ current_question + 1 }} of 10</span>
                        <span class="progress-text">Score: {{ score }}/{{ current_question }}</span>
                    </div>
                    <div class="progress-bar-container">
                        <div class="progress-bar" style="width: {{ (current_question / 10) * 100 }}%"></div>
                    </div>
                </div>
                
                <div class="question-card">
                    <div class="question-number">Question {{ current_question + 1 }}</div>
                    <div class="question-text">{{ question.question }}</div>
                    
                    <form method="POST" id="answerForm">
                        <div class="answers-grid">
                            {% for i, answer in enumerate(question.answers) %}
                            <label class="answer-option" onclick="submitAnswer('{{ answer }}')">
                                <span class="answer-letter">{{ 'ABCDEFGHIJ'[i] }}</span>
                                <span>{{ answer }}</span>
                            </label>
                            {% endfor %}
                        </div>
                        <input type="hidden" name="answer" id="selectedAnswer">
                    </form>
                </div>
                
                {% if feedback %}
                <div class="feedback {{ 'correct' if feedback.correct else 'incorrect' }}">
                    <span class="feedback-icon">{{ '🎉' if feedback.correct else '❌' }}</span>
                    <div>{{ 'Correct!' if feedback.correct else 'Incorrect!' }}</div>
                    {% if not feedback.correct %}
                    <div class="correct-answer">
                        <strong>Correct Answer:</strong> {{ feedback.correct_answer }}
                    </div>
                    {% endif %}
                    
                    <div style="margin-top: 25px;">
                        {% if current_question < 9 %}
                        <a href="/civics/{{ current_question + 1 }}" class="btn">
                            <i class="fas fa-arrow-right"></i> Next Question
                        </a>
                        {% else %}
                        <a href="/civics/results" class="btn">
                            <i class="fas fa-flag-checkered"></i> View Results
                        </a>
                        {% endif %}
                    </div>
                </div>
                {% endif %}
                
                <div style="text-align: center; margin-top: 30px;">
                    <a href="/" class="btn btn-secondary">
                        <i class="fas fa-home"></i> Back to Home
                    </a>
                </div>
            </div>
            {% endif %}
            
            {% if page == 'civics_results' %}
            <div class="test-section">
                <h2><i class="fas fa-trophy"></i> Civics Test Results</h2>
                
                <div class="final-score">
                    <div class="score-circle {{ 'pass' if passed else 'fail' }}">
                        {{ score }}/10
                    </div>
                    <h3 style="font-size: 2em; margin-bottom: 15px; color: {{ '#28a745' if passed else '#dc3545' }};">
                        {{ 'PASSED! 🎉' if passed else 'FAILED 😔' }}
                    </h3>
                    <p style="font-size: 1.3em; color: #64748b;">
                        {{ 'Congratulations! You answered enough questions correctly to pass the civics test.' if passed else 'You need to answer at least 6 questions correctly to pass. Keep studying and try again!' }}
                    </p>
                </div>
                
                <div style="background: white; padding: 30px; border-radius: 16px; margin: 30px 0; box-shadow: 0 8px 25px rgba(0,0,0,0.1);">
                    <h3 style="color: #1e293b; margin-bottom: 25px; font-size: 1.5em;">
                        <i class="fas fa-clipboard-list"></i> Detailed Review
                    </h3>
                    {% for i, result in enumerate(results) %}
                    <div style="margin: 20px 0; padding: 20px; border-left: 4px solid {{ '#28a745' if result.correct else '#dc3545' }}; background: {{ '#f8f9fa' if result.correct else '#fff5f5' }}; border-radius: 8px;">
                        <p style="font-weight: 600; margin-bottom: 10px;"><strong>Q{{ i + 1 }}:</strong> {{ result.question }}</p>
                        <p style="margin-bottom: 8px;"><strong>Your Answer:</strong> {{ result.user_answer }}</p>
                        <p style="margin-bottom: 8px;"><strong>Correct Answer:</strong> {{ result.correct_answer }}</p>
                        <p style="color: {{ '#28a745' if result.correct else '#dc3545' }}; font-weight: bold;">
                            {{ '✓ Correct' if result.correct else '✗ Incorrect' }}
                        </p>
                    </div>
                    {% endfor %}
                </div>
                
                <div style="text-align: center;">
                    <a href="/civics" class="btn">
                        <i class="fas fa-redo"></i> Take Test Again
                    </a>
                    <a href="/" class="btn btn-secondary">
                        <i class="fas fa-home"></i> Back to Home
                    </a>
                </div>
            </div>
            {% endif %}
            
            {% if page == 'english' %}
            <form method="POST">
                <div class="test-section">
                    <h2><i class="fas fa-book"></i> English Test</h2>
                    
                    <div class="question-card">
                        <h3><i class="fas fa-eye"></i> Reading Test</h3>
                        <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); border: 2px solid #2196f3; color: #0d47a1; padding: 20px; border-radius: 12px; margin-bottom: 20px;">
                            <p><strong>Instructions:</strong> Choose one sentence and read it aloud correctly. Select the sentence you can read best.</p>
                        </div>
                        {% for sentence in reading_sentences %}
                        <label class="answer-option" style="margin-bottom: 10px;">
                            <input type="radio" name="reading" value="{{ sentence }}" required style="margin-right: 15px;">
                            <span style="font-size: 1.1em;">{{ sentence }}</span>
                        </label>
                        {% endfor %}
                    </div>
                    
                    <div class="question-card">
                        <h3><i class="fas fa-pencil-alt"></i> Writing Test</h3>
                        <div style="background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%); border: 2px solid #4caf50; color: #1b5e20; padding: 20px; border-radius: 12px; margin-bottom: 20px;">
                            <p><strong>Instructions:</strong> Choose one sentence and write it exactly in the text area below.</p>
                        </div>
                        {% for sentence in writing_sentences %}
                        <label class="answer-option" style="margin-bottom: 10px;">
                            <input type="radio" name="writing_choice" value="{{ sentence }}" required onclick="document.getElementById('writing_text').placeholder='Write: ' + this.value" style="margin-right: 15px;">
                            <span style="font-size: 1.1em;">{{ sentence }}</span>
                        </label>
                        {% endfor %}
                        <textarea id="writing_text" name="writing_text" style="width: 100%; height: 120px; padding: 15px; border: 2px solid #ddd; border-radius: 12px; font-size: 16px; font-family: 'Inter', sans-serif; margin-top: 20px; resize: vertical;" placeholder="Select a sentence above, then write it here exactly as shown..." required></textarea>
                    </div>
                    
                    <div style="text-align: center; margin-top: 30px;">
                        <button type="submit" class="btn">
                            <i class="fas fa-check"></i> Submit English Test
                        </button>
                        <a href="/" class="btn btn-secondary">
                            <i class="fas fa-home"></i> Back to Home
                        </a>
                    </div>
                </div>
            </form>
            {% endif %}
            
            {% if page == 'english_result' %}
            <div class="test-section">
                <h2><i class="fas fa-chart-line"></i> English Test Results</h2>
                
                <div class="feedback {{ 'correct' if reading_passed else 'incorrect' }}" style="margin-bottom: 25px;">
                    <span class="feedback-icon">{{ '📖✓' if reading_passed else '📖✗' }}</span>
                    <h3>Reading Test: {{ 'PASSED!' if reading_passed else 'FAILED' }}</h3>
                    <p><strong>Selected Sentence:</strong> {{ reading_sentence }}</p>
                    <p>{{ 'You successfully selected a sentence to read!' if reading_passed else 'Please practice reading the vocabulary.' }}</p>
                </div>
                
                <div class="feedback {{ 'correct' if writing_passed else 'incorrect' }}" style="margin-bottom: 25px;">
                    <span class="feedback-icon">{{ '✍️✓' if writing_passed else '✍️✗' }}</span>
                    <h3>Writing Test: {{ 'PASSED!' if writing_passed else 'FAILED' }}</h3>
                    <p><strong>Target Sentence:</strong> {{ writing_target }}</p>
                    <p><strong>Your Writing:</strong> {{ writing_response }}</p>
                    <p>{{ 'Perfect! You wrote the sentence correctly.' if writing_passed else 'The sentence does not match exactly. Please try again.' }}</p>
                </div>
                
                <div class="final-score">
                    <div class="score-circle {{ 'pass' if overall_passed else 'fail' }}">
                        {{ 'PASS' if overall_passed else 'FAIL' }}
                    </div>
                    <h3 style="font-size: 2em; margin-bottom: 15px; color: {{ '#28a745' if overall_passed else '#dc3545' }};">
                        Overall: {{ 'PASSED! 🎉' if overall_passed else 'FAILED 😔' }}
                    </h3>
                    <p style="font-size: 1.3em; color: #64748b;">
                        {{ 'Congratulations! You passed both English components.' if overall_passed else 'You must pass both reading and writing to complete the English test.' }}
                    </p>
                </div>
                
                <div style="text-align: center;">
                    <a href="/english" class="btn">
                        <i class="fas fa-redo"></i> Take Test Again
                    </a>
                    <a href="/" class="btn btn-secondary">
                        <i class="fas fa-home"></i> Back to Home
                    </a>
                </div>
            </div>
            {% endif %}
        </div>
    </div>
    
    <script>
        function submitAnswer(answer) {
            document.getElementById('selectedAnswer').value = answer;
            document.getElementById('answerForm').submit();
        }
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, page='home')

@app.route('/civics')
def start_civics_test():
    # Reset test session
    session['test_questions'] = random.sample(range(1, 101), 10)
    session['current_question'] = 0
    session['score'] = 0
    session['answers'] = []
    return redirect('/civics/0')

@app.route('/civics/<int:question_num>', methods=['GET', 'POST'])
def civics_question(question_num):
    if 'test_questions' not in session:
        return redirect('/civics')
    
    if question_num >= 10:
        return redirect('/civics/results')
    
    question_id = session['test_questions'][question_num]
    question = CIVICS_QUESTIONS[question_id]
    
    feedback = None
    
    if request.method == 'POST':
        user_answer = request.form.get('answer', '').strip()
        correct_answer = question['correct']
        
        # Check if answer is correct (flexible matching)
        is_correct = False
        for possible_answer in question['answers']:
            if (user_answer.lower().strip() in possible_answer.lower() or 
                possible_answer.lower().strip() in user_answer.lower() or
                user_answer.lower().strip() == possible_answer.lower().strip()):
                is_correct = True
                break
        
        if is_correct:
            session['score'] = session.get('score', 0) + 1
        
        # Store the answer
        session['answers'] = session.get('answers', [])
        session['answers'].append({
            'question': question['question'],
            'user_answer': user_answer,
            'correct_answer': correct_answer,
            'correct': is_correct,
            'question_id': question_id
        })
        session['current_question'] = question_num + 1
        
        feedback = {
            'correct': is_correct,
            'correct_answer': correct_answer
        }
    
    return render_template_string(HTML_TEMPLATE, 
                                page='civics_question',
                                question=question,
                                current_question=question_num,
                                score=session.get('score', 0),
                                feedback=feedback)

@app.route('/civics/results')
def civics_results():
    if 'answers' not in session:
        return redirect('/civics')
    
    score = session.get('score', 0)
    passed = score >= 6
    results = session.get('answers', [])
    
    return render_template_string(HTML_TEMPLATE,
                                page='civics_results',
                                score=score,
                                passed=passed,
                                results=results)

@app.route('/english', methods=['GET', 'POST'])
def english_test():
    if request.method == 'GET':
        # Select 3 random sentences for each component
        reading_sentences = random.sample(READING_SENTENCES, 3)
        writing_sentences = random.sample(WRITING_SENTENCES, 3)
        
        session['reading_sentences'] = reading_sentences
        session['writing_sentences'] = writing_sentences
        
        return render_template_string(HTML_TEMPLATE, page='english', 
                                    reading_sentences=reading_sentences,
                                    writing_sentences=writing_sentences)
    
    else:
        # Grade the test
        reading_sentence = request.form.get('reading', '').strip()
        writing_choice = request.form.get('writing_choice', '').strip()
        writing_text = request.form.get('writing_text', '').strip()
        
        # Reading test: just need to select a sentence
        reading_passed = reading_sentence in session.get('reading_sentences', [])
        
        # Writing test: must match exactly (case insensitive, but punctuation matters)
        writing_passed = writing_choice.strip() == writing_text.strip()
        
        overall_passed = reading_passed and writing_passed
        
        return render_template_string(HTML_TEMPLATE, page='english_result',
                                    reading_passed=reading_passed,
                                    writing_passed=writing_passed,
                                    overall_passed=overall_passed,
                                    reading_sentence=reading_sentence,
                                    writing_target=writing_choice,
                                    writing_response=writing_text)

if __name__ == '__main__':
    app.run(debug=True)