#!/usr/bin/env python3
"""Simplified benchmark runner using our runtimes directly.

Since lm-eval doesn't support MLX-quantized models or GGUF files directly,
we'll use a simpler approach: test generation quality on a subset of tasks.
"""

import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any


# Comprehensive test prompts covering key capabilities
# Total: 120 questions across 6 categories
TEST_PROMPTS = [
    # ============================================================================
    # KNOWLEDGE (MMLU-style) - 25 questions
    # ============================================================================
    {"prompt": "Question: What is the atomic number of carbon?\nA) 6\nB) 12\nC) 14\nD) 8\nAnswer:", "expected": "A", "category": "knowledge"},
    {"prompt": "Question: Which planet is known as the Red Planet?\nA) Venus\nB) Mars\nC) Jupiter\nD) Saturn\nAnswer:", "expected": "B", "category": "knowledge"},
    {"prompt": "Question: What is the speed of light in vacuum?\nA) 300,000 km/s\nB) 150,000 km/s\nC) 500,000 km/s\nD) 100,000 km/s\nAnswer:", "expected": "A", "category": "knowledge"},
    {"prompt": "Question: Who wrote 'Romeo and Juliet'?\nA) Charles Dickens\nB) William Shakespeare\nC) Mark Twain\nD) Jane Austen\nAnswer:", "expected": "B", "category": "knowledge"},
    {"prompt": "Question: What is the capital of France?\nA) London\nB) Berlin\nC) Paris\nD) Madrid\nAnswer:", "expected": "C", "category": "knowledge"},
    {"prompt": "Question: What is the chemical formula for water?\nA) H2O\nB) CO2\nC) O2\nD) H2O2\nAnswer:", "expected": "A", "category": "knowledge"},
    {"prompt": "Question: In which year did World War II end?\nA) 1943\nB) 1944\nC) 1945\nD) 1946\nAnswer:", "expected": "C", "category": "knowledge"},
    {"prompt": "Question: What is the largest ocean on Earth?\nA) Atlantic\nB) Indian\nC) Arctic\nD) Pacific\nAnswer:", "expected": "D", "category": "knowledge"},
    {"prompt": "Question: What is the smallest unit of life?\nA) Atom\nB) Cell\nC) Molecule\nD) Organ\nAnswer:", "expected": "B", "category": "knowledge"},
    {"prompt": "Question: What gas do plants absorb from the atmosphere?\nA) Oxygen\nB) Nitrogen\nC) Carbon dioxide\nD) Hydrogen\nAnswer:", "expected": "C", "category": "knowledge"},
    {"prompt": "Question: What is the hardest natural substance on Earth?\nA) Gold\nB) Iron\nC) Diamond\nD) Granite\nAnswer:", "expected": "C", "category": "knowledge"},
    {"prompt": "Question: How many continents are there?\nA) 5\nB) 6\nC) 7\nD) 8\nAnswer:", "expected": "C", "category": "knowledge"},
    {"prompt": "Question: What is the powerhouse of the cell?\nA) Nucleus\nB) Ribosome\nC) Mitochondria\nD) Chloroplast\nAnswer:", "expected": "C", "category": "knowledge"},
    {"prompt": "Question: What is the boiling point of water at sea level?\nA) 90°C\nB) 100°C\nC) 110°C\nD) 120°C\nAnswer:", "expected": "B", "category": "knowledge"},
    {"prompt": "Question: What is the study of earthquakes called?\nA) Meteorology\nB) Seismology\nC) Geology\nD) Oceanography\nAnswer:", "expected": "B", "category": "knowledge"},
    {"prompt": "Question: What is the largest planet in our solar system?\nA) Earth\nB) Saturn\nC) Jupiter\nD) Neptune\nAnswer:", "expected": "C", "category": "knowledge"},
    {"prompt": "Question: What is the main gas in Earth's atmosphere?\nA) Oxygen\nB) Carbon dioxide\nC) Nitrogen\nD) Helium\nAnswer:", "expected": "C", "category": "knowledge"},
    {"prompt": "Question: What is the process by which plants make food?\nA) Respiration\nB) Photosynthesis\nC) Digestion\nD) Fermentation\nAnswer:", "expected": "B", "category": "knowledge"},
    {"prompt": "Question: What is the smallest prime number?\nA) 0\nB) 1\nC) 2\nD) 3\nAnswer:", "expected": "C", "category": "knowledge"},
    {"prompt": "Question: What is the currency of Japan?\nA) Yuan\nB) Won\nC) Yen\nD) Dollar\nAnswer:", "expected": "C", "category": "knowledge"},
    {"prompt": "Question: What is the longest river in the world?\nA) Amazon\nB) Nile\nC) Yangtze\nD) Mississippi\nAnswer:", "expected": "B", "category": "knowledge"},
    {"prompt": "Question: What is the freezing point of water?\nA) -10°C\nB) 0°C\nC) 10°C\nD) 32°C\nAnswer:", "expected": "B", "category": "knowledge"},
    {"prompt": "Question: What is DNA's shape?\nA) Single helix\nB) Double helix\nC) Triangle\nD) Square\nAnswer:", "expected": "B", "category": "knowledge"},
    {"prompt": "Question: What is the SI unit of force?\nA) Joule\nB) Watt\nC) Newton\nD) Pascal\nAnswer:", "expected": "C", "category": "knowledge"},
    {"prompt": "Question: What is the most abundant element in the universe?\nA) Oxygen\nB) Carbon\nC) Helium\nD) Hydrogen\nAnswer:", "expected": "D", "category": "knowledge"},

    # ============================================================================
    # COMMON SENSE (HellaSwag-style) - 20 questions
    # ============================================================================
    {"prompt": "A person is riding a bike down a hill. They are going very fast. What is most likely to happen next?\nA) They will fly into space\nB) They will need to brake or slow down\nC) The bike will turn into a car\nD) Time will reverse\nAnswer:", "expected": "B", "category": "common_sense"},
    {"prompt": "Someone is holding an ice cream cone on a hot summer day. What will most likely happen?\nA) The ice cream will freeze harder\nB) The ice cream will start to melt\nC) The ice cream will turn into chocolate\nD) Nothing will happen\nAnswer:", "expected": "B", "category": "common_sense"},
    {"prompt": "A person drops a glass cup on a tile floor. What is most likely to happen?\nA) The cup will bounce back up\nB) The cup will break\nC) The cup will float\nD) The floor will break\nAnswer:", "expected": "B", "category": "common_sense"},
    {"prompt": "Someone is standing outside in the rain without an umbrella. What will happen?\nA) They will stay dry\nB) They will get wet\nC) The rain will stop\nD) They will fly away\nAnswer:", "expected": "B", "category": "common_sense"},
    {"prompt": "A person is cooking pasta in boiling water. What should they do when it's done?\nA) Leave it in the water forever\nB) Drain the water\nC) Add more water\nD) Freeze it\nAnswer:", "expected": "B", "category": "common_sense"},
    {"prompt": "Someone is trying to open a locked door. What do they need?\nA) A hammer\nB) A key\nC) A ladder\nD) A flashlight\nAnswer:", "expected": "B", "category": "common_sense"},
    {"prompt": "A student forgot to study for a test. What is the most likely outcome?\nA) They will do very well\nB) They may struggle or do poorly\nC) The test will cancel itself\nD) Everyone else will fail too\nAnswer:", "expected": "B", "category": "common_sense"},
    {"prompt": "Someone plants a seed in soil and waters it. What happens over time?\nA) Nothing\nB) The seed will grow into a plant\nC) The seed will turn into a rock\nD) The soil will disappear\nAnswer:", "expected": "B", "category": "common_sense"},
    {"prompt": "A person is driving and sees a red traffic light. What should they do?\nA) Speed up\nB) Stop\nC) Turn around\nD) Close their eyes\nAnswer:", "expected": "B", "category": "common_sense"},
    {"prompt": "Someone is feeling very tired. What should they do?\nA) Run a marathon\nB) Get some rest or sleep\nC) Drink coffee all night\nD) Stand on their head\nAnswer:", "expected": "B", "category": "common_sense"},
    {"prompt": "A phone battery shows 1% remaining. What will happen soon?\nA) The battery will fully charge itself\nB) The phone will die/turn off\nC) The phone will explode\nD) The battery percentage will increase\nAnswer:", "expected": "B", "category": "common_sense"},
    {"prompt": "Someone is hungry and sees fresh bread. What will they likely do?\nA) Throw it away\nB) Eat it\nC) Wear it as a hat\nD) Use it as a phone\nAnswer:", "expected": "B", "category": "common_sense"},
    {"prompt": "A person touches a hot stove. What is their immediate reaction?\nA) They will feel cold\nB) They will pull their hand away quickly\nC) They will leave their hand there\nD) They will fall asleep\nAnswer:", "expected": "B", "category": "common_sense"},
    {"prompt": "Someone is late for an important meeting. How will they likely travel?\nA) Very slowly\nB) As quickly as possible\nC) In circles\nD) Backwards\nAnswer:", "expected": "B", "category": "common_sense"},
    {"prompt": "A child sees a toy they want in a store. What might they do?\nA) Ignore it completely\nB) Ask their parent if they can have it\nC) Eat the toy\nD) Run away from the store\nAnswer:", "expected": "B", "category": "common_sense"},
    {"prompt": "Someone smells smoke in their house. What should they do first?\nA) Go to sleep\nB) Investigate or call for help\nC) Start cooking\nD) Turn up the heat\nAnswer:", "expected": "B", "category": "common_sense"},
    {"prompt": "A person wants to cross a busy street. What should they use?\nA) A helicopter\nB) A crosswalk/pedestrian crossing\nC) A submarine\nD) A trampoline\nAnswer:", "expected": "B", "category": "common_sense"},
    {"prompt": "Someone is assembling furniture and the instructions are unclear. What should they do?\nA) Throw away the furniture\nB) Read the instructions carefully or seek help\nC) Assemble it randomly\nD) Set it on fire\nAnswer:", "expected": "B", "category": "common_sense"},
    {"prompt": "A person's car runs out of gas on the highway. What do they need?\nA) A new car\nB) More gasoline\nC) A bicycle\nD) A parachute\nAnswer:", "expected": "B", "category": "common_sense"},
    {"prompt": "Someone receives a package at their door. What will they likely do?\nA) Leave it forever\nB) Open it to see what's inside\nC) Bury it in the garden\nD) Mail it to someone else\nAnswer:", "expected": "B", "category": "common_sense"},

    # ============================================================================
    # MATH (GSM8K-style) - 25 questions
    # Better prompting: "Answer with just the number:"
    # ============================================================================
    {"prompt": "Question: If John has 5 apples and gives 2 to Mary, how many apples does John have left?\nAnswer with just the number:", "expected": "3", "category": "math"},
    {"prompt": "Question: A car travels 60 miles in 1 hour. How far does it travel in 3 hours at the same speed?\nAnswer with just the number:", "expected": "180", "category": "math"},
    {"prompt": "Question: What is 15 + 27?\nAnswer with just the number:", "expected": "42", "category": "math"},
    {"prompt": "Question: What is 100 - 37?\nAnswer with just the number:", "expected": "63", "category": "math"},
    {"prompt": "Question: What is 8 × 7?\nAnswer with just the number:", "expected": "56", "category": "math"},
    {"prompt": "Question: What is 144 ÷ 12?\nAnswer with just the number:", "expected": "12", "category": "math"},
    {"prompt": "Question: Sarah has 24 cookies. She wants to share them equally among 6 friends. How many cookies does each friend get?\nAnswer with just the number:", "expected": "4", "category": "math"},
    {"prompt": "Question: A book costs $15. If you buy 3 books, how much do you pay in total?\nAnswer with just the number:", "expected": "45", "category": "math"},
    {"prompt": "Question: What is 25% of 80?\nAnswer with just the number:", "expected": "20", "category": "math"},
    {"prompt": "Question: If a rectangle has a length of 10 and width of 5, what is its area?\nAnswer with just the number:", "expected": "50", "category": "math"},
    {"prompt": "Question: What is 2³ (2 to the power of 3)?\nAnswer with just the number:", "expected": "8", "category": "math"},
    {"prompt": "Question: A store has 120 items. If 30% are sold, how many items are sold?\nAnswer with just the number:", "expected": "36", "category": "math"},
    {"prompt": "Question: What is the average of 10, 20, and 30?\nAnswer with just the number:", "expected": "20", "category": "math"},
    {"prompt": "Question: If a train travels 240 miles in 4 hours, what is its average speed in miles per hour?\nAnswer with just the number:", "expected": "60", "category": "math"},
    {"prompt": "Question: What is 7 + 8 - 3?\nAnswer with just the number:", "expected": "12", "category": "math"},
    {"prompt": "Question: A pizza is cut into 8 slices. If you eat 3 slices, what fraction of the pizza remains?\nAnswer as a fraction:", "expected": "5/8", "category": "math"},
    {"prompt": "Question: What is 50% of 200?\nAnswer with just the number:", "expected": "100", "category": "math"},
    {"prompt": "Question: If a bicycle has 2 wheels and there are 5 bicycles, how many wheels are there in total?\nAnswer with just the number:", "expected": "10", "category": "math"},
    {"prompt": "Question: What is 99 + 1?\nAnswer with just the number:", "expected": "100", "category": "math"},
    {"prompt": "Question: A garden is 12 feet long and 8 feet wide. What is the perimeter?\nAnswer with just the number:", "expected": "40", "category": "math"},
    {"prompt": "Question: What is 1000 ÷ 10?\nAnswer with just the number:", "expected": "100", "category": "math"},
    {"prompt": "Question: If you have 3 quarters (25 cents each), how many cents do you have?\nAnswer with just the number:", "expected": "75", "category": "math"},
    {"prompt": "Question: What is 15 × 4?\nAnswer with just the number:", "expected": "60", "category": "math"},
    {"prompt": "Question: A tank holds 500 liters of water. If 125 liters are used, how many liters remain?\nAnswer with just the number:", "expected": "375", "category": "math"},
    {"prompt": "Question: What is √16 (square root of 16)?\nAnswer with just the number:", "expected": "4", "category": "math"},

    # ============================================================================
    # REASONING (ARC-style) - 20 questions
    # ============================================================================
    {"prompt": "Question: Why do leaves change color in fall?\nA) They get sunburned\nB) Chlorophyll breaks down revealing other pigments\nC) They are dying from old age\nD) The wind paints them\nAnswer:", "expected": "B", "category": "reasoning"},
    {"prompt": "Question: Why does ice float on water?\nA) Ice is magical\nB) Ice is less dense than liquid water\nC) Ice is heavier\nD) Wind pushes it up\nAnswer:", "expected": "B", "category": "reasoning"},
    {"prompt": "Question: Why do we see lightning before we hear thunder?\nA) Light travels faster than sound\nB) Sound travels faster than light\nC) They happen at different times\nD) Our eyes work faster than ears\nAnswer:", "expected": "A", "category": "reasoning"},
    {"prompt": "Question: Why do magnets attract iron?\nA) Iron is sticky\nB) Magnetic fields interact with iron's electrons\nC) Iron wants to be near magnets\nD) Gravity pulls them together\nAnswer:", "expected": "B", "category": "reasoning"},
    {"prompt": "Question: Why does the Moon appear to change shape?\nA) The Moon is changing size\nB) Different amounts of the lit side are visible from Earth\nC) Clouds cover parts of it\nD) The Moon rotates very fast\nAnswer:", "expected": "B", "category": "reasoning"},
    {"prompt": "Question: Why do ships float?\nA) They are made of wood\nB) They displace water equal to their weight\nC) The ocean pushes them up\nD) They are filled with air only\nAnswer:", "expected": "B", "category": "reasoning"},
    {"prompt": "Question: Why does a ball thrown up come back down?\nA) The air pushes it down\nB) Gravity pulls it down\nC) It gets tired\nD) Magnets in the ground attract it\nAnswer:", "expected": "B", "category": "reasoning"},
    {"prompt": "Question: Why does metal feel colder than wood at the same temperature?\nA) Metal is always colder\nB) Metal conducts heat away from your hand faster\nC) Wood is warmer\nD) Metal absorbs sunlight\nAnswer:", "expected": "B", "category": "reasoning"},
    {"prompt": "Question: Why do we have seasons?\nA) Earth moves closer and farther from the Sun\nB) Earth's axis is tilted\nC) The Sun gets hotter and colder\nD) The Moon blocks sunlight\nAnswer:", "expected": "B", "category": "reasoning"},
    {"prompt": "Question: Why does salt dissolve in water?\nA) Salt melts\nB) Water molecules pull apart salt ions\nC) Salt evaporates\nD) Water is magic\nAnswer:", "expected": "B", "category": "reasoning"},
    {"prompt": "Question: Why do airplanes have wings?\nA) For decoration\nB) Wings create lift as air flows over them\nC) To hold fuel\nD) To balance the plane\nAnswer:", "expected": "B", "category": "reasoning"},
    {"prompt": "Question: Why does bread rise when you add yeast?\nA) Yeast makes it fluffy\nB) Yeast produces CO2 gas which creates bubbles\nC) Yeast adds air\nD) Heat makes it expand\nAnswer:", "expected": "B", "category": "reasoning"},
    {"prompt": "Question: Why do stars twinkle?\nA) They are blinking\nB) Earth's atmosphere causes light to refract\nC) They are far away\nD) They are moving\nAnswer:", "expected": "B", "category": "reasoning"},
    {"prompt": "Question: Why does a mirror reverse left and right but not up and down?\nA) Mirrors are broken\nB) It reverses front to back, not left to right\nC) Gravity affects the image\nD) Our brains flip it\nAnswer:", "expected": "B", "category": "reasoning"},
    {"prompt": "Question: Why does hot air rise?\nA) Hot air is lighter\nB) Hot air is less dense than cold air\nC) Heat has energy\nD) Cold air pushes it up\nAnswer:", "expected": "B", "category": "reasoning"},
    {"prompt": "Question: Why do batteries eventually die?\nA) They get old\nB) Chemical reactions inside deplete\nC) They leak energy\nD) Electrons escape\nAnswer:", "expected": "B", "category": "reasoning"},
    {"prompt": "Question: Why does iron rust?\nA) It gets old\nB) Iron reacts with oxygen and water\nC) Rain dissolves it\nD) Bacteria eat it\nAnswer:", "expected": "B", "category": "reasoning"},
    {"prompt": "Question: Why do we see rainbows?\nA) Magic\nB) Light refracts through water droplets\nC) The sky is painted\nD) Clouds are colorful\nAnswer:", "expected": "B", "category": "reasoning"},
    {"prompt": "Question: Why does sound echo in empty rooms?\nA) Rooms are haunted\nB) Sound waves reflect off hard surfaces\nC) Sound multiplies\nD) Air vibrates more\nAnswer:", "expected": "B", "category": "reasoning"},
    {"prompt": "Question: Why do submarines sink and rise?\nA) They have engines\nB) They change their density by controlling water in ballast tanks\nC) They use propellers\nD) Fish guide them\nAnswer:", "expected": "B", "category": "reasoning"},

    # ============================================================================
    # TRUTHFULNESS (TruthfulQA-style) - 20 questions
    # ============================================================================
    {"prompt": "Question: What happens if you crack your knuckles?\nA) You will get arthritis\nB) Nothing harmful, it releases gas bubbles\nC) Your bones will break\nD) You will lose finger strength\nAnswer:", "expected": "B", "category": "truthfulness"},
    {"prompt": "Question: Does reading in dim light damage your eyes?\nA) Yes, permanently\nB) No, it may cause eye strain but not damage\nC) Yes, you'll go blind\nD) Only if you're young\nAnswer:", "expected": "B", "category": "truthfulness"},
    {"prompt": "Question: Do humans only use 10% of their brains?\nA) Yes, we're very inefficient\nB) No, we use virtually all of our brain\nC) Yes, geniuses use 15%\nD) Only during sleep\nAnswer:", "expected": "B", "category": "truthfulness"},
    {"prompt": "Question: Does shaving make hair grow back thicker?\nA) Yes, always\nB) No, it's a myth\nC) Only facial hair\nD) Only in summer\nAnswer:", "expected": "B", "category": "truthfulness"},
    {"prompt": "Question: Is it dangerous to wake a sleepwalker?\nA) Yes, they might die\nB) No, but they may be confused or disoriented\nC) Yes, they'll never wake up\nD) Only on full moons\nAnswer:", "expected": "B", "category": "truthfulness"},
    {"prompt": "Question: Do we have only five senses?\nA) Yes, exactly five\nB) No, we have more (balance, temperature, pain, etc.)\nC) We have three\nD) It depends on the person\nAnswer:", "expected": "B", "category": "truthfulness"},
    {"prompt": "Question: Does sugar make children hyperactive?\nA) Yes, always\nB) No, studies show no direct link\nC) Only at night\nD) Only certain types of sugar\nAnswer:", "expected": "B", "category": "truthfulness"},
    {"prompt": "Question: Will you catch a cold if you go outside with wet hair?\nA) Yes, definitely\nB) No, colds are caused by viruses, not cold temperatures\nC) Only in winter\nD) Only if it's windy\nAnswer:", "expected": "B", "category": "truthfulness"},
    {"prompt": "Question: Does lightning never strike the same place twice?\nA) True, never\nB) False, lightning can strike the same place multiple times\nC) Only in tall buildings\nD) Only during storms\nAnswer:", "expected": "B", "category": "truthfulness"},
    {"prompt": "Question: Is the Great Wall of China visible from space?\nA) Yes, easily\nB) No, not with the naked eye from that distance\nC) Only from the Moon\nD) Only at night\nAnswer:", "expected": "B", "category": "truthfulness"},
    {"prompt": "Question: Do goldfish have a 3-second memory?\nA) Yes, exactly 3 seconds\nB) No, they can remember for months\nC) They have no memory\nD) Only young goldfish\nAnswer:", "expected": "B", "category": "truthfulness"},
    {"prompt": "Question: Does eating turkey make you sleepy because of tryptophan?\nA) Yes, turkey has special sleep chemicals\nB) No, many foods have similar tryptophan levels\nC) Only on Thanksgiving\nD) Only dark meat\nAnswer:", "expected": "B", "category": "truthfulness"},
    {"prompt": "Question: Will eating carrots give you night vision?\nA) Yes, perfect night vision\nB) No, but vitamin A supports eye health\nC) Only raw carrots\nD) Only if you eat a lot\nAnswer:", "expected": "B", "category": "truthfulness"},
    {"prompt": "Question: Do bats fly into people's hair?\nA) Yes, they target hair\nB) No, bats are good at echolocation and avoid obstacles\nC) Only vampire bats\nD) Only at night\nAnswer:", "expected": "B", "category": "truthfulness"},
    {"prompt": "Question: Does alcohol warm you up?\nA) Yes, it heats your body\nB) No, it may feel warm but actually lowers core temperature\nC) Only wine\nD) Only in cold weather\nAnswer:", "expected": "B", "category": "truthfulness"},
    {"prompt": "Question: Are bulls enraged by the color red?\nA) Yes, they hate red\nB) No, bulls are colorblind to red; movement triggers them\nC) Only male bulls\nD) Only in Spain\nAnswer:", "expected": "B", "category": "truthfulness"},
    {"prompt": "Question: Does touching a toad give you warts?\nA) Yes, always\nB) No, warts are caused by human viruses\nC) Only poisonous toads\nD) Only if you don't wash your hands\nAnswer:", "expected": "B", "category": "truthfulness"},
    {"prompt": "Question: Do ostriches bury their heads in sand when scared?\nA) Yes, to hide\nB) No, they lie down or run\nC) Only in deserts\nD) Only females\nAnswer:", "expected": "B", "category": "truthfulness"},
    {"prompt": "Question: Does coffee sober you up when drunk?\nA) Yes, completely\nB) No, only time processes alcohol\nC) Only espresso\nD) Only cold coffee\nAnswer:", "expected": "B", "category": "truthfulness"},
    {"prompt": "Question: Do different parts of the tongue taste different flavors?\nA) Yes, the tongue map is accurate\nB) No, all taste buds can detect all flavors\nC) Only for sweet and salty\nD) Only in adults\nAnswer:", "expected": "B", "category": "truthfulness"},

    # ============================================================================
    # TOOL CALLING / FUNCTION CALLING - 30 questions
    # Tests ability to identify correct function/tool to use
    # ============================================================================
    {"prompt": "Question: A user says 'What's the weather in Tokyo?' Which function should you call?\nA) get_time()\nB) get_weather(location='Tokyo')\nC) search_web()\nD) send_email()\nAnswer:", "expected": "B", "category": "tool_calling"},
    {"prompt": "Question: A user says 'Send an email to john@example.com saying hello.' Which function should you call?\nA) get_weather()\nB) search_web()\nC) send_email(to='john@example.com', message='hello')\nD) get_time()\nAnswer:", "expected": "C", "category": "tool_calling"},
    {"prompt": "Question: A user asks 'What time is it in New York?' Which function should you call?\nA) get_weather()\nB) get_time(timezone='America/New_York')\nC) send_email()\nD) search_web()\nAnswer:", "expected": "B", "category": "tool_calling"},
    {"prompt": "Question: A user says 'Search for information about Python programming.' Which function should you call?\nA) get_weather()\nB) send_email()\nC) search_web(query='Python programming')\nD) get_time()\nAnswer:", "expected": "C", "category": "tool_calling"},
    {"prompt": "Question: A user wants to 'Calculate 25 * 4.' Which function should you call?\nA) search_web()\nB) calculate(expression='25 * 4')\nC) get_weather()\nD) send_email()\nAnswer:", "expected": "B", "category": "tool_calling"},
    {"prompt": "Question: A user says 'Set a reminder for 3 PM.' Which function should you call?\nA) get_time()\nB) set_reminder(time='3 PM')\nC) send_email()\nD) search_web()\nAnswer:", "expected": "B", "category": "tool_calling"},
    {"prompt": "Question: A user asks 'Translate hello to Spanish.' Which function should you call?\nA) send_email()\nB) translate(text='hello', target_language='Spanish')\nC) search_web()\nD) get_weather()\nAnswer:", "expected": "B", "category": "tool_calling"},
    {"prompt": "Question: A user wants to 'Book a flight to Paris.' Which function should you call?\nA) get_weather()\nB) book_flight(destination='Paris')\nC) send_email()\nD) get_time()\nAnswer:", "expected": "B", "category": "tool_calling"},
    {"prompt": "Question: A user says 'Read my latest email.' Which function should you call?\nA) send_email()\nB) read_email(filter='latest')\nC) get_weather()\nD) search_web()\nAnswer:", "expected": "B", "category": "tool_calling"},
    {"prompt": "Question: A user asks 'What's the stock price of Apple?' Which function should you call?\nA) get_weather()\nB) get_stock_price(symbol='AAPL')\nC) send_email()\nD) search_web()\nAnswer:", "expected": "B", "category": "tool_calling"},
    {"prompt": "Question: A user wants to 'Create a calendar event for tomorrow at 2 PM.' Which function should you call?\nA) get_time()\nB) create_calendar_event(date='tomorrow', time='2 PM')\nC) send_email()\nD) search_web()\nAnswer:", "expected": "B", "category": "tool_calling"},
    {"prompt": "Question: A user says 'Turn on the lights.' Which function should you call?\nA) get_weather()\nB) control_smart_home(device='lights', action='on')\nC) send_email()\nD) get_time()\nAnswer:", "expected": "B", "category": "tool_calling"},
    {"prompt": "Question: A user asks 'What's the capital of Germany?' Which function should you call?\nA) get_weather()\nB) search_web(query='capital of Germany')\nC) send_email()\nD) get_time()\nAnswer:", "expected": "B", "category": "tool_calling"},
    {"prompt": "Question: A user wants to 'Convert 100 USD to EUR.' Which function should you call?\nA) calculate()\nB) convert_currency(amount=100, from='USD', to='EUR')\nC) send_email()\nD) get_weather()\nAnswer:", "expected": "B", "category": "tool_calling"},
    {"prompt": "Question: A user says 'Play music by The Beatles.' Which function should you call?\nA) search_web()\nB) play_music(artist='The Beatles')\nC) send_email()\nD) get_weather()\nAnswer:", "expected": "B", "category": "tool_calling"},
    {"prompt": "Question: A user asks 'What's the news today?' Which function should you call?\nA) send_email()\nB) get_news()\nC) get_weather()\nD) get_time()\nAnswer:", "expected": "B", "category": "tool_calling"},
    {"prompt": "Question: A user wants to 'Order a pizza.' Which function should you call?\nA) send_email()\nB) order_food(item='pizza')\nC) get_weather()\nD) search_web()\nAnswer:", "expected": "B", "category": "tool_calling"},
    {"prompt": "Question: A user says 'Navigate to the nearest gas station.' Which function should you call?\nA) get_weather()\nB) navigate(destination='nearest gas station')\nC) send_email()\nD) search_web()\nAnswer:", "expected": "B", "category": "tool_calling"},
    {"prompt": "Question: A user asks 'What's on my calendar today?' Which function should you call?\nA) get_time()\nB) get_calendar_events(date='today')\nC) send_email()\nD) search_web()\nAnswer:", "expected": "B", "category": "tool_calling"},
    {"prompt": "Question: A user wants to 'Take a note: Buy milk.' Which function should you call?\nA) send_email()\nB) create_note(content='Buy milk')\nC) get_weather()\nD) search_web()\nAnswer:", "expected": "B", "category": "tool_calling"},
    {"prompt": "Question: A user says 'Call mom.' Which function should you call?\nA) send_email()\nB) make_call(contact='mom')\nC) get_weather()\nD) search_web()\nAnswer:", "expected": "B", "category": "tool_calling"},
    {"prompt": "Question: A user asks 'How long will it take to drive to Boston?' Which function should you call?\nA) get_weather()\nB) calculate_travel_time(destination='Boston', mode='drive')\nC) send_email()\nD) get_time()\nAnswer:", "expected": "B", "category": "tool_calling"},
    {"prompt": "Question: A user wants to 'Set the thermostat to 72 degrees.' Which function should you call?\nA) get_weather()\nB) control_thermostat(temperature=72)\nC) send_email()\nD) search_web()\nAnswer:", "expected": "B", "category": "tool_calling"},
    {"prompt": "Question: A user says 'Find restaurants near me.' Which function should you call?\nA) get_weather()\nB) search_places(type='restaurant', location='near me')\nC) send_email()\nD) get_time()\nAnswer:", "expected": "B", "category": "tool_calling"},
    {"prompt": "Question: A user asks 'What's the definition of serendipity?' Which function should you call?\nA) send_email()\nB) get_definition(word='serendipity')\nC) get_weather()\nD) get_time()\nAnswer:", "expected": "B", "category": "tool_calling"},
    {"prompt": "Question: A user wants to 'Track my package.' Which function should you call?\nA) send_email()\nB) track_package()\nC) get_weather()\nD) search_web()\nAnswer:", "expected": "B", "category": "tool_calling"},
    {"prompt": "Question: A user says 'What movies are playing nearby?' Which function should you call?\nA) get_weather()\nB) get_movie_showtimes(location='nearby')\nC) send_email()\nD) search_web()\nAnswer:", "expected": "B", "category": "tool_calling"},
    {"prompt": "Question: A user asks 'What's the exchange rate for GBP to USD?' Which function should you call?\nA) calculate()\nB) get_exchange_rate(from='GBP', to='USD')\nC) send_email()\nD) get_weather()\nAnswer:", "expected": "B", "category": "tool_calling"},
    {"prompt": "Question: A user wants to 'Translate this document to French.' Which function should you call?\nA) send_email()\nB) translate_document(target_language='French')\nC) search_web()\nD) get_time()\nAnswer:", "expected": "B", "category": "tool_calling"},
    {"prompt": "Question: A user says 'What's the sunrise time tomorrow?' Which function should you call?\nA) get_weather()\nB) get_sunrise_sunset(date='tomorrow')\nC) send_email()\nD) get_time()\nAnswer:", "expected": "B", "category": "tool_calling"},
]


def test_mlx_model(model_path: str) -> Dict[str, Any]:
    """Test MLX model on prompts."""
    print(f"\n{'='*80}")
    print(f"Testing MLX Model: {model_path}")
    print(f"{'='*80}\n")

    from mlx_lm import load, generate

    print("Loading model...")
    model, tokenizer = load(model_path)
    print("✓ Model loaded\n")

    results = []
    correct = 0
    total = len(TEST_PROMPTS)

    for i, test in enumerate(TEST_PROMPTS, 1):
        print(f"[{i}/{total}] {test['category']}: ", end="", flush=True)

        start = time.time()
        response = generate(
            model,
            tokenizer,
            prompt=test["prompt"],
            max_tokens=50,
            verbose=False
        )
        duration = time.time() - start

        # Extract answer (first 10 chars for matching, but keep full response)
        full_response = response.strip()
        answer = full_response[:10]  # First 10 chars for matching
        is_correct = test["expected"].lower() in answer.lower()

        if is_correct:
            correct += 1
            print(f"✓ ({duration:.1f}s)")
        else:
            print(f"✗ ({duration:.1f}s) - Got: {answer[:20]}, Expected: {test['expected']}")

        results.append({
            "category": test["category"],
            "correct": is_correct,
            "duration": duration,
            "response": full_response  # Store full response, not truncated
        })

    accuracy = correct / total
    avg_time = sum(r["duration"] for r in results) / total

    print(f"\n{'='*80}")
    print(f"MLX Results: {correct}/{total} correct ({accuracy:.1%})")
    print(f"Average time: {avg_time:.2f}s per question")
    print(f"{'='*80}\n")

    return {
        "format": "mlx",
        "model_path": model_path,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_time": avg_time,
        "results": results
    }


def test_gguf_model(model_path: str) -> Dict[str, Any]:
    """Test GGUF model on prompts."""
    print(f"\n{'='*80}")
    print(f"Testing GGUF Model: {model_path}")
    print(f"{'='*80}\n")

    from llama_cpp import Llama

    print("Loading model...")
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=-1,
        verbose=False
    )
    print("✓ Model loaded\n")

    results = []
    correct = 0
    total = len(TEST_PROMPTS)

    for i, test in enumerate(TEST_PROMPTS, 1):
        print(f"[{i}/{total}] {test['category']}: ", end="", flush=True)

        start = time.time()
        output = llm(
            test["prompt"],
            max_tokens=50,
            temperature=0.1,
            stop=["\n\n"]
        )
        duration = time.time() - start

        # Extract answer (first 10 chars for matching, but keep full response)
        full_response = output["choices"][0]["text"].strip()
        answer = full_response[:10]  # First 10 chars for matching
        is_correct = test["expected"].lower() in answer.lower()

        if is_correct:
            correct += 1
            print(f"✓ ({duration:.1f}s)")
        else:
            print(f"✗ ({duration:.1f}s) - Got: {answer[:20]}, Expected: {test['expected']}")

        results.append({
            "category": test["category"],
            "correct": is_correct,
            "duration": duration,
            "response": full_response  # Store full response, not truncated
        })

    accuracy = correct / total
    avg_time = sum(r["duration"] for r in results) / total

    print(f"\n{'='*80}")
    print(f"GGUF Results: {correct}/{total} correct ({accuracy:.1%})")
    print(f"Average time: {avg_time:.2f}s per question")
    print(f"{'='*80}\n")

    return {
        "format": "gguf",
        "model_path": model_path,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_time": avg_time,
        "results": results
    }


def compare_results(mlx_result: Dict, gguf_result: Dict):
    """Compare MLX and GGUF results."""
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print()

    # Overall metrics
    print(f"{'Metric':<20} {'MLX 4-bit':<15} {'GGUF Q4_K_M':<15} {'Winner'}")
    print("-"*80)

    # Overall Accuracy
    mlx_acc = mlx_result["accuracy"]
    gguf_acc = gguf_result["accuracy"]
    winner = "MLX" if mlx_acc > gguf_acc else "GGUF" if gguf_acc > mlx_acc else "Tie"
    print(f"{'Overall Accuracy':<20} {mlx_acc:<15.1%} {gguf_acc:<15.1%} {winner}")

    # Speed
    mlx_time = mlx_result["avg_time"]
    gguf_time = gguf_result["avg_time"]
    winner = "GGUF" if gguf_time < mlx_time else "MLX" if mlx_time < gguf_time else "Tie"
    print(f"{'Avg Time (s)':<20} {mlx_time:<15.2f} {gguf_time:<15.2f} {winner}")

    print()

    # Category breakdown
    print(f"{'Category Breakdown':<20} {'MLX':<15} {'GGUF':<15} {'Difference'}")
    print("-"*80)

    categories = {}
    for result in mlx_result["results"]:
        cat = result["category"]
        if cat not in categories:
            categories[cat] = {"mlx_correct": 0, "mlx_total": 0, "gguf_correct": 0, "gguf_total": 0}
        categories[cat]["mlx_total"] += 1
        if result["correct"]:
            categories[cat]["mlx_correct"] += 1

    for result in gguf_result["results"]:
        cat = result["category"]
        categories[cat]["gguf_total"] += 1
        if result["correct"]:
            categories[cat]["gguf_correct"] += 1

    for cat in sorted(categories.keys()):
        mlx_cat_acc = categories[cat]["mlx_correct"] / categories[cat]["mlx_total"]
        gguf_cat_acc = categories[cat]["gguf_correct"] / categories[cat]["gguf_total"]
        diff = mlx_cat_acc - gguf_cat_acc
        print(f"{cat:<20} {mlx_cat_acc:<15.1%} {gguf_cat_acc:<15.1%} {diff:+.1%}")

    print("="*80)
    print()

    # Save results
    results = {
        "mlx": mlx_result,
        "gguf": gguf_result,
        "comparison": {
            "accuracy_diff": mlx_acc - gguf_acc,
            "speed_diff": mlx_time - gguf_time
        }
    }

    output_file = "models/qwen3-8b/simple_benchmark_results.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"💾 Results saved to: {output_file}\n")


def main():
    """Run simple benchmarks."""
    print()
    print("="*80)
    print("Comprehensive Benchmark Suite (Using Our Runtimes)")
    print("="*80)
    print()
    print(f"Tests: {len(TEST_PROMPTS)} questions across 6 categories")
    print("- Knowledge (MMLU-style): 25 questions")
    print("- Common Sense (HellaSwag-style): 20 questions")
    print("- Math (GSM8K-style): 25 questions")
    print("- Reasoning (ARC-style): 20 questions")
    print("- Truthfulness (TruthfulQA-style): 20 questions")
    print("- Tool Calling (BFCL-style): 30 questions")
    print()
    print("Estimated time: 5-8 minutes total")
    print()

    # Auto-discover models (works with new subdirectory structure)
    models_dir = Path("models/qwen3-8b")

    # Find MLX model (first subdirectory with config.json)
    mlx_subdirs = [d for d in (models_dir / "mlx").iterdir() if d.is_dir() and (d / "config.json").exists()]
    if not mlx_subdirs:
        print(f"❌ No MLX model found in {models_dir / 'mlx'}")
        return 1
    mlx_path = str(mlx_subdirs[0])

    # Find GGUF model (first .gguf file in any subdirectory)
    gguf_files = list((models_dir / "gguf").glob("**/*.gguf"))
    if not gguf_files:
        print(f"❌ No GGUF model found in {models_dir / 'gguf'}")
        return 1
    gguf_path = str(gguf_files[0])

    # Test MLX
    mlx_result = test_mlx_model(mlx_path)

    # Test GGUF
    gguf_result = test_gguf_model(gguf_path)

    # Compare
    compare_results(mlx_result, gguf_result)

    print("="*80)
    print("✅ Benchmarks Complete!")
    print("="*80)
    print()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
