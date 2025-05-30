from flask import Flask, render_template_string, request, session, redirect, url_for
import random

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

# All 100 civics questions with multiple choice answers
QUESTIONS = [
    {
        "question": "What is the supreme law of the land?",
        "options": ["The Declaration of Independence", "The Constitution", "The Bill of Rights", "Federal Laws"],
        "correct": 1,
        "explanation": "The Constitution is the supreme law of the land."
    },
    {
        "question": "What does the Constitution do?",
        "options": ["Creates political parties", "Sets up the government", "Establishes state boundaries", "Defines citizenship"],
        "correct": 1,
        "explanation": "The Constitution sets up the government, defines the government, and protects basic rights of Americans."
    },
    {
        "question": "The idea of self-government is in the first three words of the Constitution. What are these words?",
        "options": ["We the People", "In God We Trust", "Life, Liberty, Pursuit", "United We Stand"],
        "correct": 0,
        "explanation": "The first three words of the Constitution are 'We the People'."
    },
    {
        "question": "What is an amendment?",
        "options": ["A law passed by Congress", "A change to the Constitution", "A Supreme Court decision", "A presidential order"],
        "correct": 1,
        "explanation": "An amendment is a change or addition to the Constitution."
    },
    {
        "question": "What do we call the first ten amendments to the Constitution?",
        "options": ["The Federalist Papers", "The Articles of Confederation", "The Bill of Rights", "The Declaration of Rights"],
        "correct": 2,
        "explanation": "The first ten amendments to the Constitution are called the Bill of Rights."
    },
    {
        "question": "What is one right or freedom from the First Amendment?",
        "options": ["Right to bear arms", "Freedom of speech", "Right to vote", "Right to privacy"],
        "correct": 1,
        "explanation": "The First Amendment protects freedom of speech, religion, assembly, press, and petition the government."
    },
    {
        "question": "How many amendments does the Constitution have?",
        "options": ["Twenty-five (25)", "Twenty-six (26)", "Twenty-seven (27)", "Twenty-eight (28)"],
        "correct": 2,
        "explanation": "The Constitution has twenty-seven (27) amendments."
    },
    {
        "question": "What did the Declaration of Independence do?",
        "options": ["Created the Constitution", "Announced our independence from Great Britain", "Established the Bill of Rights", "Founded the United States"],
        "correct": 1,
        "explanation": "The Declaration of Independence announced our independence from Great Britain."
    },
    {
        "question": "What are two rights in the Declaration of Independence?",
        "options": ["Life and liberty", "Freedom and justice", "Peace and prosperity", "Safety and security"],
        "correct": 0,
        "explanation": "The Declaration of Independence mentions life, liberty, and the pursuit of happiness."
    },
    {
        "question": "What is freedom of religion?",
        "options": ["You must practice Christianity", "You can practice any religion, or not practice a religion", "You must attend church weekly", "You can only practice approved religions"],
        "correct": 1,
        "explanation": "Freedom of religion means you can practice any religion, or not practice a religion."
    },
    {
        "question": "What is the economic system in the United States?",
        "options": ["Socialist economy", "Communist economy", "Capitalist economy", "Feudal economy"],
        "correct": 2,
        "explanation": "The United States has a capitalist economy or market economy."
    },
    {
        "question": "What is the 'rule of law'?",
        "options": ["Laws are suggestions", "Everyone must follow the law", "Only citizens follow laws", "Laws change daily"],
        "correct": 1,
        "explanation": "The rule of law means everyone must follow the law, including leaders and government."
    },
    {
        "question": "Name one branch or part of the government.",
        "options": ["The military", "Congress", "The police", "The FBI"],
        "correct": 1,
        "explanation": "The three branches of government are Congress (legislative), President (executive), and the courts (judicial)."
    },
    {
        "question": "What stops one branch of government from becoming too powerful?",
        "options": ["The Constitution", "Checks and balances", "The Bill of Rights", "Elections"],
        "correct": 1,
        "explanation": "Checks and balances or separation of powers stops one branch from becoming too powerful."
    },
    {
        "question": "Who is in charge of the executive branch?",
        "options": ["The Chief Justice", "The Speaker of the House", "The President", "The Senate"],
        "correct": 2,
        "explanation": "The President is in charge of the executive branch."
    },
    {
        "question": "Who makes federal laws?",
        "options": ["The President", "The Supreme Court", "Congress", "The Cabinet"],
        "correct": 2,
        "explanation": "Congress makes federal laws."
    },
    {
        "question": "What are the two parts of the U.S. Congress?",
        "options": ["House and Cabinet", "Senate and House of Representatives", "Congress and Senate", "Upper and Lower House"],
        "correct": 1,
        "explanation": "The two parts of Congress are the Senate and House of Representatives."
    },
    {
        "question": "How many U.S. Senators are there?",
        "options": ["Fifty (50)", "One hundred (100)", "Four hundred thirty-five (435)", "Ninety-nine (99)"],
        "correct": 1,
        "explanation": "There are one hundred (100) U.S. Senators."
    },
    {
        "question": "We elect a U.S. Senator for how many years?",
        "options": ["Two (2)", "Four (4)", "Six (6)", "Eight (8)"],
        "correct": 2,
        "explanation": "We elect a U.S. Senator for six (6) years."
    },
    {
        "question": "The House of Representatives has how many voting members?",
        "options": ["One hundred (100)", "Four hundred thirty-five (435)", "Five hundred (500)", "Three hundred (300)"],
        "correct": 1,
        "explanation": "The House of Representatives has four hundred thirty-five (435) voting members."
    },
    {
        "question": "We elect a U.S. Representative for how many years?",
        "options": ["Two (2)", "Four (4)", "Six (6)", "Three (3)"],
        "correct": 0,
        "explanation": "We elect a U.S. Representative for two (2) years."
    },
    {
        "question": "Who does a U.S. Senator represent?",
        "options": ["Only their political party", "All people of the state", "Only people who voted for them", "Only citizens"],
        "correct": 1,
        "explanation": "A U.S. Senator represents all people of the state."
    },
    {
        "question": "Why do some states have more Representatives than other states?",
        "options": ["Because of the state's size", "Because of the state's population", "Because of the state's age", "Because of the state's wealth"],
        "correct": 1,
        "explanation": "Some states have more Representatives because of the state's population."
    },
    {
        "question": "We elect a President for how many years?",
        "options": ["Two (2)", "Four (4)", "Six (6)", "Eight (8)"],
        "correct": 1,
        "explanation": "We elect a President for four (4) years."
    },
    {
        "question": "In what month do we vote for President?",
        "options": ["October", "November", "December", "January"],
        "correct": 1,
        "explanation": "We vote for President in November."
    },
    {
        "question": "If the President can no longer serve, who becomes President?",
        "options": ["The Speaker of the House", "The Vice President", "The Chief Justice", "The Secretary of State"],
        "correct": 1,
        "explanation": "If the President can no longer serve, the Vice President becomes President."
    },
    {
        "question": "If both the President and the Vice President can no longer serve, who becomes President?",
        "options": ["The Chief Justice", "The Secretary of State", "The Speaker of the House", "The Senate Majority Leader"],
        "correct": 2,
        "explanation": "If both the President and Vice President can no longer serve, the Speaker of the House becomes President."
    },
    {
        "question": "Who is the Commander in Chief of the military?",
        "options": ["The Secretary of Defense", "The President", "The Chief of Staff", "The Vice President"],
        "correct": 1,
        "explanation": "The President is the Commander in Chief of the military."
    },
    {
        "question": "Who signs bills to become laws?",
        "options": ["The Vice President", "The Speaker of the House", "The President", "The Chief Justice"],
        "correct": 2,
        "explanation": "The President signs bills to become laws."
    },
    {
        "question": "Who vetoes bills?",
        "options": ["The President", "The Vice President", "The Speaker of the House", "The Chief Justice"],
        "correct": 0,
        "explanation": "The President vetoes bills."
    },
    {
        "question": "What does the President's Cabinet do?",
        "options": ["Makes laws", "Advises the President", "Interprets laws", "Commands the military"],
        "correct": 1,
        "explanation": "The President's Cabinet advises the President."
    },
    {
        "question": "What does the judicial branch do?",
        "options": ["Makes laws", "Enforces laws", "Reviews laws", "Writes laws"],
        "correct": 2,
        "explanation": "The judicial branch reviews laws, explains laws, resolves disputes, and decides if a law goes against the Constitution."
    },
    {
        "question": "What is the highest court in the United States?",
        "options": ["Federal Court", "District Court", "Appeals Court", "The Supreme Court"],
        "correct": 3,
        "explanation": "The Supreme Court is the highest court in the United States."
    },
    {
        "question": "Under our Constitution, some powers belong to the federal government. What is one power of the federal government?",
        "options": ["To print money", "To give driver's licenses", "To provide police", "To provide schooling"],
        "correct": 0,
        "explanation": "Federal government powers include printing money, declaring war, creating an army, and making treaties."
    },
    {
        "question": "Under our Constitution, some powers belong to the states. What is one power of the states?",
        "options": ["To print money", "To declare war", "To provide schooling and education", "To make treaties"],
        "correct": 2,
        "explanation": "State powers include providing schooling and education, protection (police), safety (fire departments), driver's licenses, and zoning."
    },
    {
        "question": "What are the two major political parties in the United States?",
        "options": ["Republican and Libertarian", "Democratic and Republican", "Democratic and Green", "Conservative and Liberal"],
        "correct": 1,
        "explanation": "The two major political parties are Democratic and Republican."
    },
    {
        "question": "There are four amendments to the Constitution about who can vote. Describe one of them.",
        "options": ["Citizens eighteen (18) and older can vote", "Only men can vote", "Only property owners can vote", "Only educated people can vote"],
        "correct": 0,
        "explanation": "One voting amendment is that citizens eighteen (18) and older can vote."
    },
    {
        "question": "What is one responsibility that is only for United States citizens?",
        "options": ["Pay taxes", "Obey laws", "Serve on a jury", "Get an education"],
        "correct": 2,
        "explanation": "Responsibilities only for U.S. citizens include serving on a jury and voting in federal elections."
    },
    {
        "question": "Name one right only for United States citizens.",
        "options": ["Freedom of speech", "Freedom of religion", "Vote in a federal election", "Right to bear arms"],
        "correct": 2,
        "explanation": "Rights only for U.S. citizens include voting in federal elections and running for federal office."
    },
    {
        "question": "What are two rights of everyone living in the United States?",
        "options": ["Vote and run for office", "Freedom of expression and freedom of speech", "Serve on jury and vote", "Pay taxes and obey laws"],
        "correct": 1,
        "explanation": "Rights for everyone include freedom of expression, speech, assembly, petition the government, religion, and right to bear arms."
    },
    {
        "question": "What do we show loyalty to when we say the Pledge of Allegiance?",
        "options": ["The President", "The United States", "The military", "The Constitution"],
        "correct": 1,
        "explanation": "When we say the Pledge of Allegiance, we show loyalty to the United States and the flag."
    },
    {
        "question": "What is one promise you make when you become a United States citizen?",
        "options": ["To pay higher taxes", "To serve in the military for 4 years", "Give up loyalty to other countries", "To move to a different state"],
        "correct": 2,
        "explanation": "Promises include giving up loyalty to other countries, defending the Constitution, obeying laws, and serving the nation if needed."
    },
    {
        "question": "How old do citizens have to be to vote for President?",
        "options": ["Sixteen (16)", "Seventeen (17)", "Eighteen (18)", "Twenty-one (21)"],
        "correct": 2,
        "explanation": "Citizens have to be eighteen (18) and older to vote for President."
    },
    {
        "question": "What are two ways that Americans can participate in their democracy?",
        "options": ["Vote and join a political party", "Pay taxes and obey laws", "Work and study", "Drive and travel"],
        "correct": 0,
        "explanation": "Ways to participate include vote, join a political party, help with campaigns, join civic groups, contact officials, run for office."
    },
    {
        "question": "When is the last day you can send in federal income tax forms?",
        "options": ["March 15", "April 15", "May 15", "June 15"],
        "correct": 1,
        "explanation": "April 15 is the last day you can send in federal income tax forms."
    },
    {
        "question": "When must all men register for the Selective Service?",
        "options": ["At age sixteen (16)", "At age eighteen (18)", "At age twenty-one (21)", "At age twenty-five (25)"],
        "correct": 1,
        "explanation": "All men must register for the Selective Service at age eighteen (18), between eighteen and twenty-six."
    },
    {
        "question": "What is one reason colonists came to America?",
        "options": ["For gold", "For freedom", "For adventure", "For fame"],
        "correct": 1,
        "explanation": "Colonists came to America for freedom, political liberty, religious freedom, economic opportunity, and to escape persecution."
    },
    {
        "question": "Who lived in America before the Europeans arrived?",
        "options": ["Vikings", "American Indians", "Chinese", "Africans"],
        "correct": 1,
        "explanation": "American Indians (Native Americans) lived in America before the Europeans arrived."
    },
    {
        "question": "What group of people was taken to America and sold as slaves?",
        "options": ["Europeans", "Africans", "Asians", "Native Americans"],
        "correct": 1,
        "explanation": "Africans (people from Africa) were taken to America and sold as slaves."
    },
    {
        "question": "Why did the colonists fight the British?",
        "options": ["Because of religious differences", "Because of high taxes", "Because of language barriers", "Because of territorial disputes"],
        "correct": 1,
        "explanation": "Colonists fought the British because of high taxes (taxation without representation), British army staying in their houses, and lack of self-government."
    },
    {
        "question": "Who wrote the Declaration of Independence?",
        "options": ["George Washington", "Benjamin Franklin", "Thomas Jefferson", "John Adams"],
        "correct": 2,
        "explanation": "Thomas Jefferson wrote the Declaration of Independence."
    },
    {
        "question": "When was the Declaration of Independence adopted?",
        "options": ["July 4, 1775", "July 4, 1776", "July 4, 1777", "July 4, 1778"],
        "correct": 1,
        "explanation": "The Declaration of Independence was adopted on July 4, 1776."
    },
    {
        "question": "There were 13 original states. Name three.",
        "options": ["Virginia, Maryland, Georgia", "California, Texas, Florida", "Ohio, Indiana, Illinois", "Montana, Idaho, Wyoming"],
        "correct": 0,
        "explanation": "The 13 original states include New Hampshire, Massachusetts, Rhode Island, Connecticut, New York, New Jersey, Pennsylvania, Delaware, Maryland, Virginia, North Carolina, South Carolina, and Georgia."
    },
    {
        "question": "What happened at the Constitutional Convention?",
        "options": ["The Declaration of Independence was written", "The Constitution was written", "The Bill of Rights was written", "The Articles of Confederation were written"],
        "correct": 1,
        "explanation": "At the Constitutional Convention, the Constitution was written by the Founding Fathers."
    },
    {
        "question": "When was the Constitution written?",
        "options": ["1786", "1787", "1788", "1789"],
        "correct": 1,
        "explanation": "The Constitution was written in 1787."
    },
    {
        "question": "The Federalist Papers supported the passage of the U.S. Constitution. Name one of the writers.",
        "options": ["Thomas Jefferson", "James Madison", "George Washington", "Benjamin Franklin"],
        "correct": 1,
        "explanation": "The Federalist Papers were written by James Madison, Alexander Hamilton, John Jay, and Publius."
    },
    {
        "question": "What is one thing Benjamin Franklin is famous for?",
        "options": ["First President", "Writing the Constitution", "U.S. diplomat", "Leading the Continental Army"],
        "correct": 2,
        "explanation": "Benjamin Franklin was famous as a U.S. diplomat, oldest member of the Constitutional Convention, first Postmaster General, writer of 'Poor Richard's Almanac', and starting the first free libraries."
    },
    {
        "question": "Who is the 'Father of Our Country'?",
        "options": ["Thomas Jefferson", "Benjamin Franklin", "John Adams", "George Washington"],
        "correct": 3,
        "explanation": "George Washington is the 'Father of Our Country'."
    },
    {
        "question": "Who was the first President?",
        "options": ["John Adams", "Thomas Jefferson", "Benjamin Franklin", "George Washington"],
        "correct": 3,
        "explanation": "George Washington was the first President."
    },
    {
        "question": "What territory did the United States buy from France in 1803?",
        "options": ["The Florida Territory", "The Louisiana Territory", "The Texas Territory", "The California Territory"],
        "correct": 1,
        "explanation": "The United States bought the Louisiana Territory from France in 1803."
    },
    {
        "question": "Name one war fought by the United States in the 1800s.",
        "options": ["Revolutionary War", "Civil War", "World War I", "World War II"],
        "correct": 1,
        "explanation": "Wars fought by the U.S. in the 1800s include War of 1812, Mexican-American War, Civil War, and Spanish-American War."
    },
    {
        "question": "Name the U.S. war between the North and the South.",
        "options": ["Revolutionary War", "War of 1812", "Civil War", "Spanish-American War"],
        "correct": 2,
        "explanation": "The Civil War (or War between the States) was fought between the North and the South."
    },
    {
        "question": "Name one problem that led to the Civil War.",
        "options": ["Taxation", "Slavery", "Trade disputes", "Foreign wars"],
        "correct": 1,
        "explanation": "Problems that led to the Civil War include slavery, economic reasons, and states' rights."
    },
    {
        "question": "What was one important thing that Abraham Lincoln did?",
        "options": ["Won the Revolutionary War", "Wrote the Constitution", "Freed the slaves", "Bought Louisiana"],
        "correct": 2,
        "explanation": "Abraham Lincoln freed the slaves (Emancipation Proclamation), saved the Union, and led the United States during the Civil War."
    },
    {
        "question": "What did the Emancipation Proclamation do?",
        "options": ["Ended the Civil War", "Freed the slaves", "Created new states", "Established voting rights"],
        "correct": 1,
        "explanation": "The Emancipation Proclamation freed the slaves in the Confederacy and Confederate states."
    },
    {
        "question": "What did Susan B. Anthony do?",
        "options": ["Fought for women's rights", "Led the Underground Railroad", "Wrote Uncle Tom's Cabin", "Founded the Red Cross"],
        "correct": 0,
        "explanation": "Susan B. Anthony fought for women's rights and civil rights."
    },
    {
        "question": "Name one war fought by the United States in the 1900s.",
        "options": ["Civil War", "War of 1812", "World War II", "Revolutionary War"],
        "correct": 2,
        "explanation": "Wars fought by the U.S. in the 1900s include World War I, World War II, Korean War, Vietnam War, and (Persian) Gulf War."
    },
    {
        "question": "Who was President during World War I?",
        "options": ["Theodore Roosevelt", "Woodrow Wilson", "Franklin Roosevelt", "Harry Truman"],
        "correct": 1,
        "explanation": "Woodrow Wilson was President during World War I."
    },
    {
        "question": "Who was President during the Great Depression and World War II?",
        "options": ["Theodore Roosevelt", "Woodrow Wilson", "Franklin Roosevelt", "Harry Truman"],
        "correct": 2,
        "explanation": "Franklin Roosevelt was President during the Great Depression and World War II."
    },
    {
        "question": "Who did the United States fight in World War II?",
        "options": ["Germany, Italy, and Russia", "Japan, Germany, and Italy", "Japan, China, and Korea", "Germany, France, and Britain"],
        "correct": 1,
        "explanation": "The United States fought Japan, Germany, and Italy in World War II."
    },
    {
        "question": "Before he was President, Eisenhower was a general. What war was he in?",
        "options": ["World War I", "World War II", "Korean War", "Vietnam War"],
        "correct": 1,
        "explanation": "Before he was President, Eisenhower was a general in World War II."
    },
    {
        "question": "During the Cold War, what was the main concern of the United States?",
        "options": ["Fascism", "Communism", "Terrorism", "Nuclear weapons"],
        "correct": 1,
        "explanation": "During the Cold War, the main concern of the United States was Communism."
    },
    {
        "question": "What movement tried to end racial discrimination?",
        "options": ["Labor movement", "Civil rights movement", "Women's suffrage movement", "Environmental movement"],
        "correct": 1,
        "explanation": "The civil rights movement tried to end racial discrimination."
    },
    {
        "question": "What did Martin Luther King, Jr. do?",
        "options": ["Led the Underground Railroad", "Fought for civil rights", "Founded the NAACP", "Wrote the Constitution"],
        "correct": 1,
        "explanation": "Martin Luther King, Jr. fought for civil rights and worked for equality for all Americans."
    },
    {
        "question": "What major event happened on September 11, 2001, in the United States?",
        "options": ["Hurricane Katrina", "Terrorists attacked the United States", "Stock market crashed", "Presidential election"],
        "correct": 1,
        "explanation": "On September 11, 2001, terrorists attacked the United States."
    },
    {
        "question": "Name one American Indian tribe in the United States.",
        "options": ["Cherokee", "Aztec", "Inca", "Maya"],
        "correct": 0,
        "explanation": "American Indian tribes include Cherokee, Navajo, Sioux, Chippewa, Choctaw, Pueblo, Apache, Iroquois, Creek, and many others."
    },
    {
        "question": "Name one of the two longest rivers in the United States.",
        "options": ["Colorado River", "Mississippi River", "Ohio River", "Columbia River"],
        "correct": 1,
        "explanation": "The two longest rivers in the United States are the Missouri River and Mississippi River."
    },
    {
        "question": "What ocean is on the West Coast of the United States?",
        "options": ["Atlantic Ocean", "Pacific Ocean", "Arctic Ocean", "Indian Ocean"],
        "correct": 1,
        "explanation": "The Pacific Ocean is on the West Coast of the United States."
    },
    {
        "question": "What ocean is on the East Coast of the United States?",
        "options": ["Atlantic Ocean", "Pacific Ocean", "Arctic Ocean", "Indian Ocean"],
        "correct": 0,
        "explanation": "The Atlantic Ocean is on the East Coast of the United States."
    },
    {
        "question": "Name one U.S. territory.",
        "options": ["Hawaii", "Alaska", "Puerto Rico", "California"],
        "correct": 2,
        "explanation": "U.S. territories include Puerto Rico, U.S. Virgin Islands, American Samoa, Northern Mariana Islands, and Guam."
    },
    {
        "question": "Name one state that borders Canada.",
        "options": ["Texas", "Florida", "Maine", "California"],
        "correct": 2,
        "explanation": "States that border Canada include Maine, New Hampshire, Vermont, New York, Pennsylvania, Ohio, Michigan, Minnesota, North Dakota, Montana, Idaho, Washington, and Alaska."
    },
    {
        "question": "Name one state that borders Mexico.",
        "options": ["Florida", "Texas", "Louisiana", "Nevada"],
        "correct": 1,
        "explanation": "States that border Mexico include California, Arizona, New Mexico, and Texas."
    },
    {
        "question": "What is the capital of the United States?",
        "options": ["New York City", "Philadelphia", "Washington, D.C.", "Boston"],
        "correct": 2,
        "explanation": "Washington, D.C. is the capital of the United States."
    },
    {
        "question": "Where is the Statue of Liberty?",
        "options": ["Boston Harbor", "New York Harbor", "Philadelphia Harbor", "Baltimore Harbor"],
        "correct": 1,
        "explanation": "The Statue of Liberty is in New York Harbor on Liberty Island."
    },
    {
        "question": "Why does the flag have 13 stripes?",
        "options": ["Because there were 13 original colonies", "Because there were 13 founding fathers", "Because 13 is a lucky number", "Because there were 13 original laws"],
        "correct": 0,
        "explanation": "The flag has 13 stripes because there were 13 original colonies."
    },
    {
        "question": "Why does the flag have 50 stars?",
        "options": ["Because there are 50 territories", "Because there are 50 states", "Because there are 50 cities", "Because there are 50 founders"],
        "correct": 1,
        "explanation": "The flag has 50 stars because there is one star for each state (50 states)."
    },
    {
        "question": "What is the name of the national anthem?",
        "options": ["America the Beautiful", "God Bless America", "The Star-Spangled Banner", "My Country 'Tis of Thee"],
        "correct": 2,
        "explanation": "The national anthem is 'The Star-Spangled Banner'."
    },
    {
        "question": "When do we celebrate Independence Day?",
        "options": ["July 3", "July 4", "July 5", "August 4"],
        "correct": 1,
        "explanation": "We celebrate Independence Day on July 4."
    },
    {
        "question": "Name two national U.S. holidays.",
        "options": ["New Year's Day and Christmas", "Easter and Halloween", "Mother's Day and Father's Day", "Valentine's Day and St. Patrick's Day"],
        "correct": 0,
        "explanation": "National U.S. holidays include New Year's Day, Martin Luther King Jr. Day, Presidents' Day, Memorial Day, Independence Day, Labor Day, Columbus Day, Veterans Day, Thanksgiving, and Christmas."
    }
]

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>U.S. Naturalization Test Practice</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            max-width: 800px;
            width: 100%;
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 300;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .content {
            padding: 40px;
        }
        
        .progress-bar {
            background: #f0f0f0;
            border-radius: 25px;
            height: 10px;
            margin-bottom: 30px;
            overflow: hidden;
        }
        
        .progress-fill {
            background: linear-gradient(90deg, #4CAF50, #45a049);
            height: 100%;
            transition: width 0.3s ease;
        }
        
        .question-container {
            margin-bottom: 30px;
        }
        
        .question {
            font-size: 1.3em;
            font-weight: 600;
            color: #333;
            margin-bottom: 25px;
            line-height: 1.4;
        }
        
        .options {
            display: grid;
            gap: 15px;
            margin-bottom: 30px;
        }
        
        .option {
            background: #f8f9fa;
            border: 2px solid #e9ecef;
            border-radius: 12px;
            padding: 15px 20px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 1.1em;
        }
        
        .option:hover {
            background: #e9ecef;
            border-color: #6c757d;
            transform: translateY(-2px);
        }
        
        .option.selected {
            background: #007bff;
            border-color: #007bff;
            color: white;
        }
        
        .option.correct {
            background: #28a745;
            border-color: #28a745;
            color: white;
        }
        
        .option.incorrect {
            background: #dc3545;
            border-color: #dc3545;
            color: white;
        }
        
        .explanation {
            background: #e7f3ff;
            border-left: 4px solid #007bff;
            padding: 15px 20px;
            margin: 20px 0;
            border-radius: 8px;
            font-size: 1.1em;
            line-height: 1.5;
        }
        
        .explanation.correct {
            background: #d4edda;
            border-left-color: #28a745;
            color: #155724;
        }
        
        .explanation.incorrect {
            background: #f8d7da;
            border-left-color: #dc3545;
            color: #721c24;
        }
        
        .buttons {
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-top: 30px;
        }
        
        .btn {
            padding: 12px 30px;
            border: none;
            border-radius: 25px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-block;
            text-align: center;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #007bff, #0056b3);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 123, 255, 0.4);
        }
        
        .btn-success {
            background: linear-gradient(135deg, #28a745, #1e7e34);
            color: white;
        }
        
        .btn-success:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(40, 167, 69, 0.4);
        }
        
        .btn-secondary {
            background: linear-gradient(135deg, #6c757d, #545b62);
            color: white;
        }
        
        .btn-secondary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(108, 117, 125, 0.4);
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }
        
        .stat-number {
            font-size: 2.5em;
            font-weight: 700;
            color: #007bff;
            margin-bottom: 5px;
        }
        
        .stat-label {
            font-size: 1em;
            color: #6c757d;
            font-weight: 500;
        }
        
        .welcome-screen {
            text-align: center;
        }
        
        .welcome-screen h2 {
            font-size: 2em;
            color: #333;
            margin-bottom: 20px;
        }
        
        .welcome-screen p {
            font-size: 1.2em;
            color: #666;
            margin-bottom: 30px;
            line-height: 1.6;
        }
        
        .results-screen {
            text-align: center;
        }
        
        .results-screen h2 {
            font-size: 2.5em;
            margin-bottom: 20px;
        }
        
        .results-screen h2.pass {
            color: #28a745;
        }
        
        .results-screen h2.fail {
            color: #dc3545;
        }
        
        .final-score {
            font-size: 1.5em;
            margin-bottom: 30px;
            color: #333;
        }
        
        @media (max-width: 768px) {
            .container {
                margin: 10px;
            }
            
            .header {
                padding: 20px;
            }
            
            .header h1 {
                font-size: 2em;
            }
            
            .content {
                padding: 20px;
            }
            
            .question {
                font-size: 1.1em;
            }
            
            .option {
                padding: 12px 15px;
                font-size: 1em;
            }
            
            .buttons {
                flex-direction: column;
            }
            
            .btn {
                width: 100%;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🇺🇸 U.S. Naturalization Test</h1>
            <p>Practice for Your Citizenship Test</p>
        </div>
        
        <div class="content">
            {% if not session.get('started') %}
                <div class="welcome-screen">
                    <h2>Welcome to the Naturalization Test Practice</h2>
                    <p>This practice test will help you prepare for the civics portion of the U.S. naturalization test. You'll be asked 10 random questions from the official list of 100 civics questions.</p>
                    <p>To pass the actual test, you need to answer 6 out of 10 questions correctly. Good luck!</p>
                    <div class="buttons">
                        <a href="{{ url_for('start_test') }}" class="btn btn-primary">Start Practice Test</a>
                    </div>
                </div>
            {% elif session.get('completed') %}
                <div class="results-screen">
                    <h2 class="{% if session.score >= 6 %}pass{% else %}fail{% endif %}">
                        {% if session.score >= 6 %}
                            🎉 Congratulations!
                        {% else %}
                            📚 Keep Studying!
                        {% endif %}
                    </h2>
                    <div class="final-score">
                        You scored {{ session.score }} out of {{ session.total_questions }} questions correctly.
                    </div>
                    <div class="stats">
                        <div class="stat-card">
                            <div class="stat-number">{{ session.score }}</div>
                            <div class="stat-label">Correct Answers</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">{{ ((session.score / session.total_questions) * 100)|round|int }}%</div>
                            <div class="stat-label">Success Rate</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">{{ session.total_questions - session.score }}</div>
                            <div class="stat-label">Incorrect Answers</div>
                        </div>
                    </div>
                    <p style="margin-bottom: 30px; font-size: 1.1em; color: #666;">
                        {% if session.score >= 6 %}
                            Great job! You would pass the actual naturalization test. Keep practicing to maintain your knowledge.
                        {% else %}
                            You need at least 6 correct answers to pass. Review the study materials and try again.
                        {% endif %}
                    </p>
                    <div class="buttons">
                        <a href="{{ url_for('restart_test') }}" class="btn btn-primary">Take Another Test</a>
                        <a href="{{ url_for('index') }}" class="btn btn-secondary">Back to Home</a>
                    </div>
                </div>
            {% else %}
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {{ ((session.current_question + 1) / session.total_questions * 100)|round }}%"></div>
                </div>
                
                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-number">{{ session.current_question + 1 }}</div>
                        <div class="stat-label">Current Question</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{{ session.score }}</div>
                        <div class="stat-label">Correct So Far</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{{ session.total_questions - session.current_question - 1 }}</div>
                        <div class="stat-label">Remaining</div>
                    </div>
                </div>
                
                {% if session.get('show_result') %}
                    <div class="question-container">
                        <div class="question">{{ current_question.question }}</div>
                        <div class="options">
                            {% for i, option in enumerate(current_question.options) %}
                                <div class="option 
                                    {% if i == session.selected_answer %}
                                        {% if i == current_question.correct %}
                                            correct
                                        {% else %}
                                            incorrect
                                        {% endif %}
                                    {% elif i == current_question.correct %}
                                        correct
                                    {% endif %}">
                                    {{ option }}
                                </div>
                            {% endfor %}
                        </div>
                        <div class="explanation {% if session.selected_answer == current_question.correct %}correct{% else %}incorrect{% endif %}">
                            {% if session.selected_answer == current_question.correct %}
                                ✅ Correct! {{ current_question.explanation }}
                            {% else %}
                                ❌ Incorrect. {{ current_question.explanation }}
                            {% endif %}
                        </div>
                        <div class="buttons">
                            <a href="{{ url_for('next_question') }}" class="btn btn-success">
                                {% if session.current_question + 1 >= session.total_questions %}
                                    View Results
                                {% else %}
                                    Next Question
                                {% endif %}
                            </a>
                        </div>
                    </div>
                {% else %}
                    <form method="POST" action="{{ url_for('answer_question') }}">
                        <div class="question-container">
                            <div class="question">{{ current_question.question }}</div>
                            <div class="options">
                                {% for i, option in enumerate(current_question.options) %}
                                    <label>
                                        <input type="radio" name="answer" value="{{ i }}" style="display: none;" onchange="selectOption(this, {{ i }})">
                                        <div class="option" onclick="selectOption(this.previousElementSibling, {{ i }})">
                                            {{ option }}
                                        </div>
                                    </label>
                                {% endfor %}
                            </div>
                            <div class="buttons">
                                <button type="submit" class="btn btn-primary" id="submit-btn" disabled>Submit Answer</button>
                            </div>
                        </div>
                    </form>
                {% endif %}
            {% endif %}
        </div>
    </div>

    <script>
        function selectOption(radio, index) {
            // Remove previous selections
            document.querySelectorAll('.option').forEach(opt => {
                opt.classList.remove('selected');
            });
            
            // Add selection to clicked option
            radio.nextElementSibling.classList.add('selected');
            radio.checked = true;
            
            // Enable submit button
            document.getElementById('submit-btn').disabled = false;
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    session.clear()
    return render_template_string(HTML_TEMPLATE)

@app.route('/start')
def start_test():
    # Initialize session with random 10 questions
    selected_questions = random.sample(QUESTIONS, 10)
    session['questions'] = selected_questions
    session['current_question'] = 0
    session['score'] = 0
    session['total_questions'] = 10
    session['started'] = True
    session['completed'] = False
    session['show_result'] = False
    return redirect(url_for('question'))

@app.route('/question')
def question():
    if not session.get('started') or session.get('completed'):
        return redirect(url_for('index'))
    
    current_q = session['questions'][session['current_question']]
    return render_template_string(HTML_TEMPLATE, current_question=current_q)

@app.route('/answer', methods=['POST'])
def answer_question():
    if not session.get('started') or session.get('completed'):
        return redirect(url_for('index'))
    
    selected_answer = int(request.form.get('answer', -1))
    current_q = session['questions'][session['current_question']]
    
    session['selected_answer'] = selected_answer
    session['show_result'] = True
    
    if selected_answer == current_q['correct']:
        session['score'] += 1
    
    return redirect(url_for('question'))

@app.route('/next')
def next_question():
    if not session.get('started'):
        return redirect(url_for('index'))
    
    session['current_question'] += 1
    session['show_result'] = False
    
    if session['current_question'] >= session['total_questions']:
        session['completed'] = True
        return redirect(url_for('results'))
    
    return redirect(url_for('question'))

@app.route('/results')
def results():
    if not session.get('completed'):
        return redirect(url_for('index'))
    
    return render_template_string(HTML_TEMPLATE)

@app.route('/restart')
def restart_test():
    session.clear()
    return redirect(url_for('start_test'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)