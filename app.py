from flask import Flask, request, session, jsonify, render_template_string
import random
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# ALL 100 CIVICS QUESTIONS
CIVICS_QUESTIONS = [
    {
        "question": "What is the supreme law of the land?",
        "options": ["the Constitution", "the Declaration of Independence", "the Bill of Rights", "Federal Laws"],
        "correct": 0,
        "explanation": "The Constitution is the supreme law of the land, establishing the framework for government and protecting basic rights."
    },
    {
        "question": "What does the Constitution do?",
        "options": ["sets up the government", "declares independence", "lists all federal laws", "creates the military"],
        "correct": 0,
        "explanation": "The Constitution sets up the government, defines the government, and protects basic rights of Americans."
    },
    {
        "question": "The idea of self-government is in the first three words of the Constitution. What are these words?",
        "options": ["We the People", "In God We Trust", "Life, Liberty, Pursuit", "All Men Created"],
        "correct": 0,
        "explanation": "The Constitution begins with 'We the People,' emphasizing that government derives its power from the citizens."
    },
    {
        "question": "What is an amendment?",
        "options": ["a change (to the Constitution)", "a law", "a court decision", "a presidential order"],
        "correct": 0,
        "explanation": "An amendment is a change or addition to the Constitution."
    },
    {
        "question": "What do we call the first ten amendments to the Constitution?",
        "options": ["the Bill of Rights", "the Articles of Confederation", "the Constitutional Amendments", "the Founding Principles"],
        "correct": 0,
        "explanation": "The first ten amendments are called the Bill of Rights, protecting fundamental freedoms."
    },
    {
        "question": "What is one right or freedom from the First Amendment?",
        "options": ["speech", "right to bear arms", "right to vote", "right to privacy"],
        "correct": 0,
        "explanation": "The First Amendment protects freedom of speech, religion, press, assembly, and petition."
    },
    {
        "question": "How many amendments does the Constitution have?",
        "options": ["twenty-seven (27)", "ten (10)", "twenty-five (25)", "thirty-three (33)"],
        "correct": 0,
        "explanation": "The Constitution has 27 amendments, including the original Bill of Rights."
    },
    {
        "question": "What did the Declaration of Independence do?",
        "options": ["announced our independence (from Great Britain)", "created the Constitution", "established the Bill of Rights", "formed the first government"],
        "correct": 0,
        "explanation": "The Declaration of Independence announced that the American colonies were free from British control."
    },
    {
        "question": "What are two rights in the Declaration of Independence?",
        "options": ["life, liberty", "freedom and justice", "peace and prosperity", "honor and duty"],
        "correct": 0,
        "explanation": "The Declaration mentions the rights to life, liberty, and the pursuit of happiness."
    },
    {
        "question": "What is freedom of religion?",
        "options": ["You can practice any religion, or not practice a religion", "You must practice Christianity", "You cannot practice religion", "You must choose one religion"],
        "correct": 0,
        "explanation": "Freedom of religion means you can practice any religion or choose not to practice a religion."
    },
    {
        "question": "What is the economic system in the United States?",
        "options": ["capitalist economy", "socialist economy", "communist economy", "feudal economy"],
        "correct": 0,
        "explanation": "The United States has a capitalist or market economy based on free enterprise."
    },
    {
        "question": "What is the 'rule of law'?",
        "options": ["Everyone must follow the law", "Only citizens follow laws", "Laws are suggestions", "Leaders make all laws"],
        "correct": 0,
        "explanation": "The rule of law means everyone, including leaders and government, must obey the law. No one is above the law."
    },
    {
        "question": "Name one branch or part of the government.",
        "options": ["Congress", "Federal", "State", "Local"],
        "correct": 0,
        "explanation": "The three branches are legislative (Congress), executive (President), and judicial (courts)."
    },
    {
        "question": "What stops one branch of government from becoming too powerful?",
        "options": ["checks and balances", "the Constitution", "the President", "Congress"],
        "correct": 0,
        "explanation": "Checks and balances and separation of powers ensure each branch can limit the power of the other branches."
    },
    {
        "question": "Who is in charge of the executive branch?",
        "options": ["the President", "Congress", "the Supreme Court", "the Speaker of the House"],
        "correct": 0,
        "explanation": "The President leads the executive branch and enforces federal laws."
    },
    {
        "question": "Who makes federal laws?",
        "options": ["Congress", "the President", "the Supreme Court", "State governments"],
        "correct": 0,
        "explanation": "Congress, consisting of the Senate and House of Representatives, makes federal laws."
    },
    {
        "question": "What are the two parts of the U.S. Congress?",
        "options": ["the Senate and House (of Representatives)", "Upper and Lower House", "Federal and State Congress", "Democratic and Republican"],
        "correct": 0,
        "explanation": "Congress consists of the Senate and the House of Representatives."
    },
    {
        "question": "How many U.S. Senators are there?",
        "options": ["one hundred (100)", "fifty (50)", "four hundred thirty-five (435)", "five hundred thirty-eight (538)"],
        "correct": 0,
        "explanation": "There are 100 U.S. Senators - two from each of the 50 states."
    },
    {
        "question": "We elect a U.S. Senator for how many years?",
        "options": ["six (6)", "two (2)", "four (4)", "eight (8)"],
        "correct": 0,
        "explanation": "U.S. Senators serve six-year terms."
    },
    {
        "question": "Who is one of your state's U.S. Senators now?",
        "options": ["Answers will vary", "The Governor", "The Mayor", "The President"],
        "correct": 0,
        "explanation": "Each state has two U.S. Senators. You should know who represents your state."
    },
    {
        "question": "The House of Representatives has how many voting members?",
        "options": ["four hundred thirty-five (435)", "one hundred (100)", "five hundred thirty-eight (538)", "three hundred fifty (350)"],
        "correct": 0,
        "explanation": "The House of Representatives has 435 voting members, representing districts based on population."
    },
    {
        "question": "We elect a U.S. Representative for how many years?",
        "options": ["two (2)", "four (4)", "six (6)", "three (3)"],
        "correct": 0,
        "explanation": "U.S. Representatives serve two-year terms."
    },
    {
        "question": "Name your U.S. Representative.",
        "options": ["Answers will vary", "The Senator", "The Governor", "The Mayor"],
        "correct": 0,
        "explanation": "You should know the name of your U.S. Representative who represents your congressional district."
    },
    {
        "question": "Who does a U.S. Senator represent?",
        "options": ["all people of the state", "only voters", "only citizens", "only adults"],
        "correct": 0,
        "explanation": "A U.S. Senator represents all people living in their state."
    },
    {
        "question": "Why do some states have more Representatives than other states?",
        "options": ["(because of) the state's population", "because they are larger", "because they are older", "because they have more money"],
        "correct": 0,
        "explanation": "States with larger populations have more Representatives in the House."
    },
    {
        "question": "We elect a President for how many years?",
        "options": ["four (4)", "two (2)", "six (6)", "eight (8)"],
        "correct": 0,
        "explanation": "The President serves a four-year term and can be re-elected once."
    },
    {
        "question": "In what month do we vote for President?",
        "options": ["November", "October", "December", "September"],
        "correct": 0,
        "explanation": "Presidential elections are held in November, on the first Tuesday after the first Monday."
    },
    {
        "question": "What is the name of the President of the United States now?",
        "options": ["Visit uscis.gov/citizenship/testupdates", "Donald Trump", "Joe Biden", "Barack Obama"],
        "correct": 0,
        "explanation": "Check the USCIS website for the current President's name at the time of your test."
    },
    {
        "question": "What is the name of the Vice President of the United States now?",
        "options": ["Visit uscis.gov/citizenship/testupdates", "Kamala Harris", "Mike Pence", "Joe Biden"],
        "correct": 0,
        "explanation": "Check the USCIS website for the current Vice President's name at the time of your test."
    },
    {
        "question": "If the President can no longer serve, who becomes President?",
        "options": ["the Vice President", "Speaker of the House", "Secretary of State", "Chief Justice"],
        "correct": 0,
        "explanation": "The Vice President becomes President if the President cannot serve."
    },
    {
        "question": "If both the President and the Vice President can no longer serve, who becomes President?",
        "options": ["the Speaker of the House", "Secretary of State", "Chief Justice", "Senate Majority Leader"],
        "correct": 0,
        "explanation": "The Speaker of the House is next in line after the Vice President."
    },
    {
        "question": "Who is the Commander in Chief of the military?",
        "options": ["the President", "Secretary of Defense", "Chairman of Joint Chiefs", "General of the Army"],
        "correct": 0,
        "explanation": "The President serves as the Commander in Chief of all U.S. armed forces."
    },
    {
        "question": "Who signs bills to become laws?",
        "options": ["the President", "Speaker of the House", "Chief Justice", "Senate Majority Leader"],
        "correct": 0,
        "explanation": "The President signs bills passed by Congress to make them federal laws."
    },
    {
        "question": "Who vetoes bills?",
        "options": ["the President", "Congress", "Supreme Court", "Vice President"],
        "correct": 0,
        "explanation": "The President has the power to veto (reject) bills passed by Congress."
    },
    {
        "question": "What does the President's Cabinet do?",
        "options": ["advises the President", "makes laws", "interprets laws", "enforces state laws"],
        "correct": 0,
        "explanation": "The President's Cabinet consists of department heads who advise the President."
    },
    {
        "question": "What are two Cabinet-level positions?",
        "options": ["Secretary of Defense, Secretary of State", "Speaker and Majority Leader", "Chief Justice and Associate Justice", "Senator and Representative"],
        "correct": 0,
        "explanation": "Cabinet positions include Secretary of Defense, State, Treasury, and many others."
    },
    {
        "question": "What does the judicial branch do?",
        "options": ["reviews laws", "makes laws", "enforces laws", "votes on laws"],
        "correct": 0,
        "explanation": "The judicial branch reviews laws, explains laws, and decides if laws go against the Constitution."
    },
    {
        "question": "What is the highest court in the United States?",
        "options": ["the Supreme Court", "Federal Court", "District Court", "Appeals Court"],
        "correct": 0,
        "explanation": "The Supreme Court is the highest court and final authority on constitutional questions."
    },
    {
        "question": "How many justices are on the Supreme Court?",
        "options": ["Visit uscis.gov/citizenship/testupdates", "seven (7)", "eight (8)", "ten (10)"],
        "correct": 0,
        "explanation": "Check the USCIS website for the current number of Supreme Court justices."
    },
    {
        "question": "Who is the Chief Justice of the United States now?",
        "options": ["Visit uscis.gov/citizenship/testupdates", "John Roberts", "Clarence Thomas", "Ruth Bader Ginsburg"],
        "correct": 0,
        "explanation": "Check the USCIS website for the current Chief Justice's name."
    },
    {
        "question": "Under our Constitution, some powers belong to the federal government. What is one power of the federal government?",
        "options": ["to print money", "to give driver's licenses", "to provide schooling", "to provide police"],
        "correct": 0,
        "explanation": "Federal powers include printing money, declaring war, creating an army, and making treaties."
    },
    {
        "question": "Under our Constitution, some powers belong to the states. What is one power of the states?",
        "options": ["provide schooling and education", "print money", "declare war", "make treaties"],
        "correct": 0,
        "explanation": "State powers include providing education, protection (police), safety (fire departments), driver's licenses, and zoning."
    },
    {
        "question": "Who is the Governor of your state now?",
        "options": ["Answers will vary", "The President", "The Mayor", "The Senator"],
        "correct": 0,
        "explanation": "You should know who is the current Governor of your state."
    },
    {
        "question": "What is the capital of your state?",
        "options": ["Answers will vary", "Washington D.C.", "New York", "Los Angeles"],
        "correct": 0,
        "explanation": "You should know the capital city of your state."
    },
    {
        "question": "What are the two major political parties in the United States?",
        "options": ["Democratic and Republican", "Liberal and Conservative", "Federal and State", "Progressive and Traditional"],
        "correct": 0,
        "explanation": "The Democratic and Republican parties are the two major political parties."
    },
    {
        "question": "What is the political party of the President now?",
        "options": ["Visit uscis.gov/citizenship/testupdates", "Democratic", "Republican", "Independent"],
        "correct": 0,
        "explanation": "Check the USCIS website for the current President's political party."
    },
    {
        "question": "What is the name of the Speaker of the House of Representatives now?",
        "options": ["Visit uscis.gov/citizenship/testupdates", "Nancy Pelosi", "Kevin McCarthy", "Paul Ryan"],
        "correct": 0,
        "explanation": "Check the USCIS website for the current Speaker of the House."
    },
    {
        "question": "There are four amendments to the Constitution about who can vote. Describe one of them.",
        "options": ["Citizens eighteen (18) and older (can vote)", "Only men can vote", "Only property owners can vote", "Only educated people can vote"],
        "correct": 0,
        "explanation": "The 26th Amendment allows citizens 18 and older to vote. Other amendments eliminated poll taxes and expanded voting rights."
    },
    {
        "question": "What is one responsibility that is only for United States citizens?",
        "options": ["serve on a jury", "pay taxes", "obey laws", "attend school"],
        "correct": 0,
        "explanation": "Only U.S. citizens can serve on juries and vote in federal elections."
    },
    {
        "question": "Name one right only for United States citizens.",
        "options": ["vote in a federal election", "freedom of speech", "freedom of religion", "right to bear arms"],
        "correct": 0,
        "explanation": "Only citizens can vote in federal elections and run for federal office."
    },
    {
        "question": "What are two rights of everyone living in the United States?",
        "options": ["freedom of expression, freedom of speech", "right to vote, right to run for office", "right to a job, right to housing", "right to free education, right to healthcare"],
        "correct": 0,
        "explanation": "Everyone in the U.S. has freedom of expression, speech, assembly, petition, religion, and the right to bear arms."
    },
    {
        "question": "What do we show loyalty to when we say the Pledge of Allegiance?",
        "options": ["the United States", "the President", "Congress", "the military"],
        "correct": 0,
        "explanation": "The Pledge of Allegiance shows loyalty to the United States and the flag."
    },
    {
        "question": "What is one promise you make when you become a United States citizen?",
        "options": ["give up loyalty to other countries", "never leave the United States", "pay higher taxes", "serve in the military for 4 years"],
        "correct": 0,
        "explanation": "New citizens promise to give up loyalty to other countries, defend the Constitution, obey laws, and serve the nation if needed."
    },
    {
        "question": "How old do citizens have to be to vote for President?",
        "options": ["eighteen (18) and older", "twenty-one (21) and older", "sixteen (16) and older", "twenty-five (25) and older"],
        "correct": 0,
        "explanation": "Citizens must be 18 years or older to vote in federal elections."
    },
    {
        "question": "What are two ways that Americans can participate in their democracy?",
        "options": ["vote, join a political party", "pay taxes, obey laws", "work, go to school", "drive, own property"],
        "correct": 0,
        "explanation": "Americans can participate by voting, joining political parties, campaigning, contacting officials, running for office, and more."
    },
    {
        "question": "When is the last day you can send in federal income tax forms?",
        "options": ["April 15", "March 15", "May 15", "December 31"],
        "correct": 0,
        "explanation": "Federal income tax forms are due on April 15th each year."
    },
    {
        "question": "When must all men register for the Selective Service?",
        "options": ["at age eighteen (18)", "at age twenty-one (21)", "at age sixteen (16)", "when they vote"],
        "correct": 0,
        "explanation": "All men must register for Selective Service at age 18, between ages 18 and 26."
    },
    {
        "question": "What is one reason colonists came to America?",
        "options": ["freedom", "to escape war", "for better weather", "to find gold"],
        "correct": 0,
        "explanation": "Colonists came for freedom, political liberty, religious freedom, economic opportunity, and to escape persecution."
    },
    {
        "question": "Who lived in America before the Europeans arrived?",
        "options": ["American Indians", "Spanish explorers", "French traders", "Russian settlers"],
        "correct": 0,
        "explanation": "American Indians (Native Americans) lived in America before European colonization."
    },
    {
        "question": "What group of people was taken to America and sold as slaves?",
        "options": ["Africans", "Europeans", "Asians", "Native Americans"],
        "correct": 0,
        "explanation": "Africans were taken to America and sold as slaves, mainly to work on plantations."
    },
    {
        "question": "Why did the colonists fight the British?",
        "options": ["because of high taxes (taxation without representation)", "religious differences", "land disputes", "trade restrictions only"],
        "correct": 0,
        "explanation": "Colonists fought because of high taxes, British army staying in their houses, and lack of self-government."
    },
    {
        "question": "Who wrote the Declaration of Independence?",
        "options": ["(Thomas) Jefferson", "George Washington", "Benjamin Franklin", "John Adams"],
        "correct": 0,
        "explanation": "Thomas Jefferson was the primary author of the Declaration of Independence."
    },
    {
        "question": "When was the Declaration of Independence adopted?",
        "options": ["July 4, 1776", "July 4, 1775", "May 4, 1776", "September 4, 1776"],
        "correct": 0,
        "explanation": "The Declaration of Independence was adopted on July 4, 1776."
    },
    {
        "question": "There were 13 original states. Name three.",
        "options": ["New York, New Jersey, Pennsylvania", "California, Texas, Florida", "Ohio, Michigan, Illinois", "Washington, Oregon, Montana"],
        "correct": 0,
        "explanation": "The 13 original states include New Hampshire, Massachusetts, Rhode Island, Connecticut, New York, New Jersey, Pennsylvania, Delaware, Maryland, Virginia, North Carolina, South Carolina, and Georgia."
    },
    {
        "question": "What happened at the Constitutional Convention?",
        "options": ["The Constitution was written", "Independence was declared", "The first President was chosen", "The Bill of Rights was created"],
        "correct": 0,
        "explanation": "At the Constitutional Convention in 1787, the Founding Fathers wrote the Constitution."
    },
    {
        "question": "When was the Constitution written?",
        "options": ["1787", "1776", "1783", "1791"],
        "correct": 0,
        "explanation": "The Constitution was written in 1787 during the Constitutional Convention in Philadelphia."
    },
    {
        "question": "The Federalist Papers supported the passage of the U.S. Constitution. Name one of the writers.",
        "options": ["(James) Madison", "George Washington", "Thomas Jefferson", "John Hancock"],
        "correct": 0,
        "explanation": "The Federalist Papers were written by James Madison, Alexander Hamilton, and John Jay (Publius)."
    },
    {
        "question": "What is one thing Benjamin Franklin is famous for?",
        "options": ["U.S. diplomat", "First President", "Writing Declaration", "Leading the army"],
        "correct": 0,
        "explanation": "Benjamin Franklin was a U.S. diplomat, oldest member of Constitutional Convention, first Postmaster General, writer of Poor Richard's Almanac, and started first free libraries."
    },
    {
        "question": "Who is the 'Father of Our Country'?",
        "options": ["(George) Washington", "Thomas Jefferson", "Benjamin Franklin", "John Adams"],
        "correct": 0,
        "explanation": "George Washington is called the 'Father of Our Country' for his leadership during the Revolution and as first President."
    },
    {
        "question": "Who was the first President?",
        "options": ["(George) Washington", "John Adams", "Thomas Jefferson", "Benjamin Franklin"],
        "correct": 0,
        "explanation": "George Washington was the first President of the United States (1789-1797)."
    },
    {
        "question": "What territory did the United States buy from France in 1803?",
        "options": ["the Louisiana Territory", "Florida Territory", "Oregon Territory", "Texas Territory"],
        "correct": 0,
        "explanation": "The Louisiana Purchase in 1803 doubled the size of the United States."
    },
    {
        "question": "Name one war fought by the United States in the 1800s.",
        "options": ["War of 1812", "World War I", "World War II", "Korean War"],
        "correct": 0,
        "explanation": "Wars in the 1800s include War of 1812, Mexican-American War, Civil War, and Spanish-American War."
    },
    {
        "question": "Name the U.S. war between the North and the South.",
        "options": ["the Civil War", "Revolutionary War", "War of 1812", "Mexican-American War"],
        "correct": 0,
        "explanation": "The Civil War (1861-1865) was fought between the North and South."
    },
    {
        "question": "Name one problem that led to the Civil War.",
        "options": ["slavery", "taxes", "trade", "immigration"],
        "correct": 0,
        "explanation": "The Civil War was caused by slavery, economic reasons, and states' rights."
    },
    {
        "question": "What was one important thing that Abraham Lincoln did?",
        "options": ["freed the slaves (Emancipation Proclamation)", "wrote the Constitution", "led the Revolutionary War", "bought Louisiana"],
        "correct": 0,
        "explanation": "Abraham Lincoln freed the slaves, saved the Union, and led during the Civil War."
    },
    {
        "question": "What did the Emancipation Proclamation do?",
        "options": ["freed the slaves", "ended the Civil War", "created new states", "established voting rights"],
        "correct": 0,
        "explanation": "The Emancipation Proclamation freed slaves in the Confederacy and Confederate states."
    },
    {
        "question": "What did Susan B. Anthony do?",
        "options": ["fought for women's rights", "was the first female President", "led the Underground Railroad", "founded the Red Cross"],
        "correct": 0,
        "explanation": "Susan B. Anthony fought for women's rights and civil rights."
    },
    {
        "question": "Name one war fought by the United States in the 1900s.",
        "options": ["World War I", "Civil War", "Revolutionary War", "War of 1812"],
        "correct": 0,
        "explanation": "Wars in the 1900s include World War I, World War II, Korean War, Vietnam War, and Gulf War."
    },
    {
        "question": "Who was President during World War I?",
        "options": ["(Woodrow) Wilson", "Theodore Roosevelt", "Franklin Roosevelt", "Harry Truman"],
        "correct": 0,
        "explanation": "Woodrow Wilson was President during World War I (1917-1918)."
    },
    {
        "question": "Who was President during the Great Depression and World War II?",
        "options": ["(Franklin) Roosevelt", "Theodore Roosevelt", "Harry Truman", "Dwight Eisenhower"],
        "correct": 0,
        "explanation": "Franklin D. Roosevelt was President during the Great Depression and most of World War II."
    },
    {
        "question": "Who did the United States fight in World War II?",
        "options": ["Japan, Germany, and Italy", "Russia, China, and Korea", "Britain, France, and Spain", "Mexico, Canada, and Cuba"],
        "correct": 0,
        "explanation": "The U.S. fought against the Axis powers: Japan, Germany, and Italy in World War II."
    },
    {
        "question": "Before he was President, Eisenhower was a general. What war was he in?",
        "options": ["World War II", "World War I", "Korean War", "Vietnam War"],
        "correct": 0,
        "explanation": "Dwight Eisenhower was a general in World War II before becoming President."
    },
    {
        "question": "During the Cold War, what was the main concern of the United States?",
        "options": ["Communism", "Fascism", "Monarchy", "Anarchy"],
        "correct": 0,
        "explanation": "During the Cold War, the main U.S. concern was the spread of Communism."
    },
    {
        "question": "What movement tried to end racial discrimination?",
        "options": ["civil rights (movement)", "labor movement", "suffrage movement", "temperance movement"],
        "correct": 0,
        "explanation": "The civil rights movement tried to end racial discrimination."
    },
    {
        "question": "What did Martin Luther King, Jr. do?",
        "options": ["fought for civil rights", "led the Civil War", "was a President", "wrote the Constitution"],
        "correct": 0,
        "explanation": "Martin Luther King, Jr. fought for civil rights and worked for equality for all Americans."
    },
    {
        "question": "What major event happened on September 11, 2001, in the United States?",
        "options": ["Terrorists attacked the United States", "The U.S. declared war", "A major earthquake occurred", "The President was inaugurated"],
        "correct": 0,
        "explanation": "On September 11, 2001, terrorists attacked the World Trade Center and Pentagon."
    },
    {
        "question": "Name one American Indian tribe in the United States.",
        "options": ["Cherokee", "Vikings", "Celts", "Saxons"],
        "correct": 0,
        "explanation": "American Indian tribes include Cherokee, Navajo, Sioux, Chippewa, Choctaw, Pueblo, Apache, Iroquois, and many others."
    },
    {
        "question": "Name one of the two longest rivers in the United States.",
        "options": ["Missouri (River)", "Colorado River", "Hudson River", "Rio Grande"],
        "correct": 0,
        "explanation": "The Missouri River and Mississippi River are the two longest rivers in the United States."
    },
    {
        "question": "What ocean is on the West Coast of the United States?",
        "options": ["Pacific (Ocean)", "Atlantic Ocean", "Indian Ocean", "Arctic Ocean"],
        "correct": 0,
        "explanation": "The Pacific Ocean borders the West Coast of the United States."
    },
    {
        "question": "What ocean is on the East Coast of the United States?",
        "options": ["Atlantic (Ocean)", "Pacific Ocean", "Indian Ocean", "Arctic Ocean"],
        "correct": 0,
        "explanation": "The Atlantic Ocean borders the East Coast of the United States."
    },
    {
        "question": "Name one U.S. territory.",
        "options": ["Puerto Rico", "Hawaii", "Alaska", "Washington D.C."],
        "correct": 0,
        "explanation": "U.S. territories include Puerto Rico, U.S. Virgin Islands, American Samoa, Northern Mariana Islands, and Guam."
    },
    {
        "question": "Name one state that borders Canada.",
        "options": ["Maine", "California", "Texas", "Florida"],
        "correct": 0,
        "explanation": "States bordering Canada include Maine, New Hampshire, Vermont, New York, Pennsylvania, Ohio, Michigan, Minnesota, North Dakota, Montana, Idaho, Washington, and Alaska."
    },
    {
        "question": "Name one state that borders Mexico.",
        "options": ["California", "Nevada", "Colorado", "Oklahoma"],
        "correct": 0,
        "explanation": "States bordering Mexico are California, Arizona, New Mexico, and Texas."
    },
    {
        "question": "What is the capital of the United States?",
        "options": ["Washington, D.C.", "New York City", "Philadelphia", "Boston"],
        "correct": 0,
        "explanation": "Washington, D.C. is the capital of the United States."
    },
    {
        "question": "Where is the Statue of Liberty?",
        "options": ["New York (Harbor)", "Boston Harbor", "San Francisco Bay", "Chesapeake Bay"],
        "correct": 0,
        "explanation": "The Statue of Liberty is located in New York Harbor on Liberty Island."
    },
    {
        "question": "Why does the flag have 13 stripes?",
        "options": ["because there were 13 original colonies", "for the 13 founding fathers", "for 13 years of war", "for 13 amendments"],
        "correct": 0,
        "explanation": "The 13 stripes represent the 13 original colonies that declared independence from Britain."
    },
    {
        "question": "Why does the flag have 50 stars?",
        "options": ["because there is one star for each state", "for 50 years of independence", "for 50 founding fathers", "for 50 amendments"],
        "correct": 0,
        "explanation": "The 50 stars represent the 50 states of the United States."
    },
    {
        "question": "What is the name of the national anthem?",
        "options": ["The Star-Spangled Banner", "America the Beautiful", "God Bless America", "My Country 'Tis of Thee"],
        "correct": 0,
        "explanation": "The Star-Spangled Banner is the national anthem of the United States."
    },
    {
        "question": "When do we celebrate Independence Day?",
        "options": ["July 4", "July 14", "May 4", "September 4"],
        "correct": 0,
        "explanation": "Independence Day is celebrated on July 4th, commemorating the adoption of the Declaration of Independence."
    },
    {
        "question": "Name two national U.S. holidays.",
        "options": ["New Year's Day, Christmas", "Valentine's Day, Halloween", "Earth Day, Arbor Day", "Easter, Passover"],
        "correct": 0,
        "explanation": "National holidays include New Year's Day, Martin Luther King Jr. Day, Presidents' Day, Memorial Day, Independence Day, Labor Day, Columbus Day, Veterans Day, Thanksgiving, and Christmas."
    }
]

# ENGLISH READING VOCABULARY
READING_VOCABULARY = [
    # PEOPLE
    "Abraham Lincoln", "George Washington",
    # CIVICS
    "American flag", "Bill of Rights", "capital", "citizen", "city", "Congress", "country", 
    "Father of Our Country", "government", "President", "right", "Senators", "state", "states", "White House",
    # PLACES
    "America", "United States", "U.S.",
    # HOLIDAYS
    "Presidents' Day", "Memorial Day", "Flag Day", "Independence Day", "Labor Day", "Columbus Day", "Thanksgiving",
    # QUESTION WORDS
    "How", "What", "When", "Where", "Who", "Why",
    # VERBS
    "can", "come", "do", "does", "elects", "have", "has", "is", "are", "was", "be", "lives", "lived", "meet", "name", "pay", "vote", "want",
    # OTHER FUNCTION
    "a", "for", "here", "in", "of", "on", "the", "to", "we",
    # OTHER CONTENT
    "colors", "dollar bill", "first", "largest", "many", "most", "north", "one", "people", "second", "south"
]

# ENGLISH WRITING VOCABULARY  
WRITING_VOCABULARY = [
    # PEOPLE
    "Adams", "Lincoln", "Washington",
    # CIVICS
    "American Indians", "capital", "citizens", "Civil War", "Congress", "Father of Our Country", "flag", "free", 
    "freedom of speech", "President", "right", "Senators", "state", "states", "White House",
    # PLACES
    "Alaska", "California", "Canada", "Delaware", "Mexico", "New York City", "United States", "Washington", "Washington, D.C.",
    # MONTHS
    "February", "May", "June", "July", "September", "October", "November",
    # HOLIDAYS
    "Presidents' Day", "Memorial Day", "Flag Day", "Independence Day", "Labor Day", "Columbus Day", "Thanksgiving",
    # VERBS
    "can", "come", "elect", "have", "has", "is", "was", "be", "lives", "lived", "meets", "pay", "vote", "want",
    # OTHER FUNCTION
    "and", "during", "for", "here", "in", "of", "on", "the", "to", "we",
    # OTHER CONTENT
    "blue", "dollar bill", "fifty", "50", "first", "largest", "most", "north", "one", "one hundred", "100", 
    "people", "red", "second", "south", "taxes", "white"
]

# ENGLISH READING TEST SENTENCES
READING_SENTENCES = [
    "America is the land of freedom.",
    "All people want to be free.",
    "America is the home of the brave.",
    "America is the land of the free.",
    "Citizens have the right to freedom of speech.",
    "All citizens have the right to vote.",
    "George Washington was the first President.",
    "Abraham Lincoln was President during the Civil War.",
    "The President lives in the White House.",
    "The President is the Commander in Chief.",
    "Congress meets in the capital.",
    "Congress makes the laws.",
    "The capital of the United States is Washington, D.C.",
    "Washington, D.C. is the capital of the United States.",
    "Citizens have many rights.",
    "Citizens have the right to vote.",
    "George Washington is the Father of Our Country.",
    "Martha Washington was the first First Lady.",
    "The American flag has stars and stripes.",
    "The flag has thirteen stripes.",
    "The Statue of Liberty was a gift from France.",
    "France gave the Statue of Liberty to America.",
    "Independence Day is in July.",
    "The Fourth of July is a holiday.",
    "Labor Day is in September.",
    "Memorial Day is in May.",
    "Presidents' Day is in February.",
    "Thanksgiving is in November.",
    "Columbus Day is in October.",
    "Flag Day is in June."
]

# ENGLISH WRITING TEST SENTENCES
WRITING_SENTENCES = [
    "Washington was the first President.",
    "Adams was the second President.",
    "Lincoln was President during the Civil War.",
    "George Washington is the Father of Our Country.",
    "The President lives in the White House.",
    "The White House is in Washington, D.C.",
    "Washington, D.C., is the capital of the United States.",
    "The capital of the United States is Washington, D.C.",
    "New York City has the most people.",
    "Alaska is the largest state.",
    "California is south of Canada.",
    "Delaware is north of Mexico.",
    "Canada is north of the United States.",
    "Mexico is south of the United States.",
    "Citizens can vote.",
    "Citizens have the right to vote.",
    "All citizens can vote.",
    "Only citizens can vote.",
    "Citizens vote for the President in November.",
    "We vote for President in November.",
    "We vote for Congress in November.",
    "Congress meets for the country.",
    "Congress makes laws for the country.",
    "All people want to be free.",
    "All people want freedom.",
    "America is the land of the free.",
    "America is free.",
    "America has freedom of speech.",
    "Everyone in America has freedom of speech.",
    "American Indians lived here first.",
    "American Indians came to America first.",
    "The American flag has red, white, and blue.",
    "The flag is red, white, and blue.",
    "The American flag has fifty stars.",
    "The flag has fifty stars.",
    "There are fifty states in America.",
    "America has fifty states.",
    "Independence Day is July 4.",
    "The Fourth of July is Independence Day.",
    "We celebrate Independence Day in July.",
    "Memorial Day is in May.",
    "Presidents' Day is in February.",
    "Labor Day is in September.",
    "Columbus Day is in October.",
    "Flag Day is in June.",
    "Thanksgiving is in November.",
    "Citizens pay taxes.",
    "People in America pay taxes.",
    "All people pay taxes.",
    "Lincoln lived in the White House.",
    "Washington lived in the White House.",
    "Adams lived in the White House.",
    "The President has lived in the White House for one hundred years.",
    "People come to America for freedom.",
    "People come to America to be free.",
    "People come to America for a better life."
]

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Complete U.S. Naturalization Test Practice</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Georgia', serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #333;
        }
        
        .container {
            background: white;
            border-radius: 15px;
            padding: 2rem;
            max-width: 900px;
            width: 90%;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            position: relative;
            overflow: hidden;
        }
        
        .container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 5px;
            background: linear-gradient(90deg, #ff6b35, #f7931e, #ffd23f);
        }
        
        .header {
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .header h1 {
            color: #1e3c72;
            font-size: 2.2rem;
            margin-bottom: 0.5rem;
        }
        
        .header p {
            color: #666;
            font-size: 1.1rem;
        }
        
        .test-selection {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin: 2rem 0;
        }
        
        .test-card {
            background: #f8f9fa;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            padding: 1.5rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .test-card:hover {
            border-color: #2196f3;
            transform: translateY(-3px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .test-card h3 {
            color: #1e3c72;
            margin-bottom: 1rem;
        }
        
        .test-card p {
            color: #666;
            margin-bottom: 1rem;
        }
        
        .progress-bar {
            background: #e0e0e0;
            height: 8px;
            border-radius: 4px;
            margin: 1rem 0 2rem 0;
            overflow: hidden;
        }
        
        .progress {
            background: linear-gradient(90deg, #4CAF50, #45a049);
            height: 100%;
            transition: width 0.3s ease;
        }
        
        .score-board {
            display: flex;
            justify-content: space-between;
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 2rem;
        }
        
        .score-item {
            text-align: center;
        }
        
        .score-number {
            font-size: 1.5rem;
            font-weight: bold;
            color: #1e3c72;
        }
        
        .score-label {
            font-size: 0.9rem;
            color: #666;
        }
        
        .question-card {
            background: #fff;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .question-number {
            color: #1e3c72;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        
        .question-text {
            font-size: 1.3rem;
            line-height: 1.5;
            margin-bottom: 1.5rem;
            color: #333;
        }
        
        .options {
            display: grid;
            gap: 1rem;
        }
        
        .option {
            background: #f8f9fa;
            border: 2px solid #e0e0e0;
            padding: 1rem;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 1.1rem;
        }
        
        .option:hover {
            background: #e3f2fd;
            border-color: #2196f3;
            transform: translateY(-2px);
        }
        
        .option.correct {
            background: #e8f5e8;
            border-color: #4caf50;
            color: #2e7d32;
        }
        
        .option.incorrect {
            background: #ffebee;
            border-color: #f44336;
            color: #c62828;
        }
        
        .explanation {
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 1rem;
            margin-top: 1rem;
            border-radius: 0 8px 8px 0;
        }
        
        .explanation h4 {
            color: #1565c0;
            margin-bottom: 0.5rem;
        }
        
        .reading-test, .writing-test {
            text-align: center;
        }
        
        .sentence-display {
            background: #f8f9fa;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            padding: 2rem;
            margin: 2rem 0;
            font-size: 1.4rem;
            line-height: 1.6;
            color: #333;
        }
        
        .writing-input {
            width: 100%;
            padding: 1rem;
            font-size: 1.2rem;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            margin: 1rem 0;
            font-family: 'Georgia', serif;
        }
        
        .writing-input:focus {
            outline: none;
            border-color: #2196f3;
        }
        
        .buttons {
            display: flex;
            gap: 1rem;
            justify-content: center;
            margin-top: 2rem;
            flex-wrap: wrap;
        }
        
        .btn {
            background: #1e3c72;
            color: white;
            border: none;
            padding: 1rem 2rem;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1.1rem;
            transition: all 0.3s ease;
        }
        
        .btn:hover {
            background: #2a5298;
            transform: translateY(-2px);
        }
        
        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        
        .btn.secondary {
            background: #6c757d;
        }
        
        .btn.secondary:hover {
            background: #5a6268;
        }
        
        .final-score {
            text-align: center;
            padding: 2rem;
        }
        
        .final-score h2 {
            color: #1e3c72;
            font-size: 2rem;
            margin-bottom: 1rem;
        }
        
        .final-score .score {
            font-size: 3rem;
            font-weight: bold;
            color: #4caf50;
            margin: 1rem 0;
        }
        
        .final-score .message {
            font-size: 1.2rem;
            color: #666;
            margin-bottom: 2rem;
        }
        
        .citizenship-info {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 1.5rem;
            margin-top: 1rem;
            border-left: 4px solid #ff6b35;
        }
        
        .citizenship-info h3 {
            color: #1e3c72;
            margin-bottom: 1rem;
        }
        
        .citizenship-info ul {
            text-align: left;
            margin: 1rem 0;
        }
        
        .english-result {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            border-left: 4px solid #4caf50;
        }
        
        .english-result.incorrect {
            border-left-color: #f44336;
        }
        
        @media (max-width: 600px) {
            .container {
                padding: 1.5rem;
                margin: 1rem;
            }
            
            .header h1 {
                font-size: 1.8rem;
            }
            
            .question-text {
                font-size: 1.1rem;
            }
            
            .buttons {
                flex-direction: column;
            }
            
            .score-board {
                flex-direction: column;
                gap: 1rem;
            }
            
            .test-selection {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        {% if not session.get('test_started') %}
        <div class="header">
            <h1>🇺🇸 Complete U.S. Naturalization Test Practice</h1>
            <p>Master ALL components of the naturalization test</p>
        </div>
        
        <div class="citizenship-info">
            <h3>About the Naturalization Test</h3>
            <p>The naturalization test has TWO main components:</p>
            <ul>
                <li><strong>English Test:</strong> Reading, Writing, and Speaking</li>
                <li><strong>Civics Test:</strong> U.S. History and Government (6 out of 10 questions correct)</li>
            </ul>
            <p>This practice includes ALL 100 official civics questions and complete English vocabulary!</p>
        </div>
        
        <div class="test-selection">
            <div class="test-card" onclick="startTest('civics')">
                <h3>📚 Civics Test</h3>
                <p>Practice all 100 official civics questions about U.S. government and history</p>
                <strong>10 random questions • Need 6 correct to pass</strong>
            </div>
            
            <div class="test-card" onclick="startTest('reading')">
                <h3>📖 English Reading Test</h3>
                <p>Practice reading sentences with official vocabulary words</p>
                <strong>Read 1 out of 3 sentences correctly</strong>
            </div>
            
            <div class="test-card" onclick="startTest('writing')">
                <h3>✍️ English Writing Test</h3>
                <p>Practice writing sentences with official vocabulary words</p>
                <strong>Write 1 out of 3 sentences correctly</strong>
            </div>
        </div>
        
        {% elif session.get('test_completed') %}
        <div class="final-score">
            <h2>🎉 Test Complete!</h2>
            {% if session.get('test_type') == 'civics' %}
                <div class="score">{{ session.get('correct_answers', 0) }}/{{ session.get('total_questions', 10) }}</div>
                <div class="message">
                    {% if session.get('correct_answers', 0) >= 6 %}
                        Excellent! You passed the civics test! 🎊<br>
                        You're well-prepared for the actual naturalization interview.
                    {% else %}
                        Keep studying! You need at least 6 correct answers to pass.<br>
                        Review the questions you missed and try again.
                    {% endif %}
                </div>
            {% elif session.get('test_type') == 'reading' %}
                <div class="message">
                    {% if session.get('reading_passed') %}
                        Perfect! You passed the reading test! 📖<br>
                        You successfully read the required sentence.
                    {% else %}
                        Keep practicing! You need to read at least one sentence correctly.<br>
                        Study the vocabulary words and try again.
                    {% endif %}
                </div>
            {% elif session.get('test_type') == 'writing' %}
                <div class="message">
                    {% if session.get('writing_passed') %}
                        Excellent! You passed the writing test! ✍️<br>
                        You successfully wrote the required sentence.
                    {% else %}
                        Keep practicing! You need to write at least one sentence correctly.<br>
                        Study the vocabulary words and try again.
                    {% endif %}
                </div>
            {% endif %}
            
            <div class="citizenship-info">
                <h3>Next Steps Toward U.S. Citizenship</h3>
                <p>Ready to apply for naturalization? Visit <strong>uscis.gov</strong> to:</p>
                <ul>
                    <li>Check your eligibility requirements</li>
                    <li>Download Form N-400 (Application for Naturalization)</li>
                    <li>Find additional study materials and resources</li>
                    <li>Schedule your naturalization interview</li>
                    <li>Learn about the Oath of Allegiance ceremony</li>
                </ul>
            </div>
            
            <div class="buttons">
                <button class="btn" onclick="goHome()">Take Another Test</button>
                <button class="btn secondary" onclick="startTest('civics')" {% if session.get('test_type') == 'civics' %}style="display:none"{% endif %}>Try Civics Test</button>
                <button class="btn secondary" onclick="startTest('reading')" {% if session.get('test_type') == 'reading' %}style="display:none"{% endif %}>Try Reading Test</button>
                <button class="btn secondary" onclick="startTest('writing')" {% if session.get('test_type') == 'writing' %}style="display:none"{% endif %}>Try Writing Test</button>
            </div>
        </div>
        
        {% elif session.get('test_type') == 'civics' %}
        <div class="header">
            <h1>📚 Civics Test Practice</h1>
            <p>U.S. History and Government</p>
        </div>
        
        <div class="progress-bar">
            <div class="progress" style="width: {{ (session.get('current_question', 0) / session.get('total_questions', 10)) * 100 }}%"></div>
        </div>
        
        <div class="score-board">
            <div class="score-item">
                <div class="score-number">{{ session.get('current_question', 0) }}</div>
                <div class="score-label">Question</div>
            </div>
            <div class="score-item">
                <div class="score-number">{{ session.get('correct_answers', 0) }}</div>
                <div class="score-label">Correct</div>
            </div>
            <div class="score-item">
                <div class="score-number">{{ session.get('total_questions', 10) - session.get('current_question', 0) }}</div>
                <div class="score-label">Remaining</div>
            </div>
        </div>
        
        {% if current_question %}
        <div class="question-card">
            <div class="question-number">Question {{ session.get('current_question', 0) }} of {{ session.get('total_questions', 10) }}</div>
            <div class="question-text">{{ current_question.question }}</div>
            
            <div class="options" id="options">
                {% for i, option in enumerate(current_question.options) %}
                <div class="option" onclick="selectAnswer({{ i }})" id="option-{{ i }}">
                    {{ option }}
                </div>
                {% endfor %}
            </div>
            
            <div id="explanation" style="display: none;">
                <div class="explanation">
                    <h4>Explanation:</h4>
                    <p>{{ current_question.explanation }}</p>
                </div>
            </div>
        </div>
        
        <div class="buttons">
            <button class="btn" id="nextBtn" onclick="nextQuestion()" style="display: none;">Next Question</button>
        </div>
        {% endif %}
        
        {% elif session.get('test_type') == 'reading' %}
        <div class="header">
            <h1>📖 English Reading Test</h1>
            <p>Read the sentence aloud clearly</p>
        </div>
        
        <div class="reading-test">
            {% if session.get('reading_sentences') %}
                <div class="question-number">Sentence {{ session.get('reading_attempt', 0) + 1 }} of 3</div>
                <div class="sentence-display">
                    {{ session.get('reading_sentences')[session.get('reading_attempt', 0)] }}
                </div>
                <p style="margin: 1rem 0;">Read this sentence aloud. In the actual test, the USCIS officer will listen to you read.</p>
                
                <div class="buttons">
                    <button class="btn" onclick="readingSuccess()">I Read It Correctly</button>
                    <button class="btn secondary" onclick="readingFailed()">I Need to Try Another Sentence</button>
                </div>
            {% endif %}
        </div>
        
        {% elif session.get('test_type') == 'writing' %}
        <div class="header">
            <h1>✍️ English Writing Test</h1>
            <p>Write the sentence exactly as shown</p>
        </div>
        
        <div class="writing-test">
            {% if session.get('writing_sentences') %}
                <div class="question-number">Sentence {{ session.get('writing_attempt', 0) + 1 }} of 3</div>
                <div class="sentence-display">
                    {{ session.get('writing_sentences')[session.get('writing_attempt', 0)] }}
                </div>
                <p style="margin: 1rem 0;">Write this sentence in the box below:</p>
                <input type="text" class="writing-input" id="writingInput" placeholder="Type the sentence here...">
                
                <div class="buttons">
                    <button class="btn" onclick="checkWriting()">Check My Writing</button>
                </div>
                
                <div id="writingResult" style="display: none;"></div>
            {% endif %}
        </div>
        {% endif %}
        
        <div class="buttons" style="margin-top: 1rem;">
            <button class="btn secondary" onclick="goHome()">← Back to Test Selection</button>
        </div>
    </div>

    <script>
        function startTest(testType) {
            fetch('/start', { 
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ test_type: testType })
            }).then(() => location.reload());
        }
        
        function selectAnswer(selectedIndex) {
            const options = document.querySelectorAll('.option');
            const explanation = document.getElementById('explanation');
            const nextBtn = document.getElementById('nextBtn');
            
            options.forEach(option => {
                option.style.pointerEvents = 'none';
            });
            
            fetch('/answer', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ answer: selectedIndex })
            })
            .then(response => response.json())
            .then(data => {
                options[data.correct_answer].classList.add('correct');
                
                if (selectedIndex !== data.correct_answer) {
                    options[selectedIndex].classList.add('incorrect');
                }
                
                explanation.style.display = 'block';
                nextBtn.style.display = 'inline-block';
            });
        }
        
        function nextQuestion() {
            fetch('/next', { method: 'POST' })
                .then(() => location.reload());
        }
        
        function readingSuccess() {
            fetch('/reading_result', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ success: true })
            }).then(() => location.reload());
        }
        
        function readingFailed() {
            fetch('/reading_result', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ success: false })
            }).then(() => location.reload());
        }
        
        function checkWriting() {
            const input = document.getElementById('writingInput').value;
            const result = document.getElementById('writingResult');
            
            fetch('/check_writing', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ written_text: input })
            })
            .then(response => response.json())
            .then(data => {
                result.style.display = 'block';
                if (data.correct) {
                    result.innerHTML = '<div class="english-result"><h4>✅ Correct!</h4><p>Perfect! You wrote the sentence correctly.</p><button class="btn" onclick="location.reload()">Complete Test</button></div>';
                } else {
                    result.innerHTML = '<div class="english-result incorrect"><h4>❌ Try Again</h4><p>Not quite right. Check your spelling, capitalization, and punctuation.</p><p><strong>Expected:</strong> ' + data.expected + '</p><p><strong>You wrote:</strong> ' + input + '</p><button class="btn secondary" onclick="tryAgainWriting()">Try Another Sentence</button></div>';
                }
            });
        }
        
        function tryAgainWriting() {
            fetch('/writing_next', { method: 'POST' })
                .then(() => location.reload());
        }
        
        function goHome() {
            fetch('/restart', { method: 'POST' })
                .then(() => location.reload());
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    current_question = None
    if session.get('test_started') and not session.get('test_completed') and session.get('test_type') == 'civics':
        question_index = session.get('current_question_index')
        if question_index is not None:
            current_question = session.get('questions')[question_index]
    
    return render_template_string(HTML_TEMPLATE, 
                                current_question=current_question,
                                enumerate=enumerate)

@app.route('/start', methods=['POST'])
def start_test():
    data = request.get_json()
    test_type = data.get('test_type', 'civics')
    
    # Reset session
    session.clear()
    session['test_started'] = True
    session['test_completed'] = False
    session['test_type'] = test_type
    
    if test_type == 'civics':
        # Select 10 random civics questions
        selected_questions = random.sample(CIVICS_QUESTIONS, 10)
        session['questions'] = selected_questions
        session['current_question'] = 1
        session['current_question_index'] = 0
        session['total_questions'] = 10
        session['correct_answers'] = 0
    
    elif test_type == 'reading':
        # Select 3 random reading sentences
        selected_sentences = random.sample(READING_SENTENCES, 3)
        session['reading_sentences'] = selected_sentences
        session['reading_attempt'] = 0
        session['reading_passed'] = False
        
    elif test_type == 'writing':
        # Select 3 random writing sentences
        selected_sentences = random.sample(WRITING_SENTENCES, 3)
        session['writing_sentences'] = selected_sentences
        session['writing_attempt'] = 0
        session['writing_passed'] = False
    
    return '', 200

@app.route('/answer', methods=['POST'])
def check_answer():
    data = request.get_json()
    selected_answer = data.get('answer')
    
    current_question_index = session.get('current_question_index')
    current_question = session.get('questions')[current_question_index]
    
    is_correct = selected_answer == current_question['correct']
    
    if is_correct:
        session['correct_answers'] = session.get('correct_answers', 0) + 1
    
    return jsonify({
        'correct': is_correct,
        'correct_answer': current_question['correct']
    })

@app.route('/next', methods=['POST'])
def next_question():
    current_question = session.get('current_question', 1)
    total_questions = session.get('total_questions', 10)
    
    if current_question >= total_questions:
        session['test_completed'] = True
    else:
        session['current_question'] = current_question + 1
        session['current_question_index'] = session.get('current_question_index', 0) + 1
    
    return '', 200

@app.route('/reading_result', methods=['POST'])
def reading_result():
    data = request.get_json()
    success = data.get('success', False)
    
    if success:
        session['reading_passed'] = True
        session['test_completed'] = True
    else:
        attempt = session.get('reading_attempt', 0)
        if attempt >= 2:  # Failed all 3 attempts
            session['test_completed'] = True
        else:
            session['reading_attempt'] = attempt + 1
    
    return '', 200

@app.route('/check_writing', methods=['POST'])
def check_writing():
    data = request.get_json()
    written_text = data.get('written_text', '').strip()
    
    attempt = session.get('writing_attempt', 0)
    expected_sentence = session.get('writing_sentences')[attempt]
    
    # Simple comparison (case-insensitive, basic punctuation)
    is_correct = written_text.lower().replace('.', '').replace(',', '') == expected_sentence.lower().replace('.', '').replace(',', '')
    
    if is_correct:
        session['writing_passed'] = True
        session['test_completed'] = True
    
    return jsonify({
        'correct': is_correct,
        'expected': expected_sentence
    })

@app.route('/writing_next', methods=['POST'])
def writing_next():
    attempt = session.get('writing_attempt', 0)
    if attempt >= 2:  # Failed all 3 attempts
        session['test_completed'] = True
    else:
        session['writing_attempt'] = attempt + 1
    
    return '', 200

@app.route('/restart', methods=['POST'])
def restart_test():
    session.clear()
    return '', 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
