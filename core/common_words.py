"""core/common_words.py — the English lexicon behind core/topic_hygiene.py.

Used ONLY to tell a real phrase ("garden shed", "temperature conversion")
from one made mostly of non-words — the typical shape of a Whisper
mis-transcription of TV or room audio that got learned as a "topic".

``lexicon()`` is the full known-word set, built LAZILY on first use (the
learner and the audit tool call it; importing this module costs nothing):

  * ``core/english_words.txt`` — ~19k lowercase English words derived from
    the GPT-2 byte-pair vocabulary bundled with OpenAI Whisper. MIT licence;
    the full notice and how the list was cut are in that file's header.
  * ``COMMON_WORDS`` below — a hand-written everyday-word list (base forms and
    irregular forms), written for this project.
  * ``PLACES`` — countries, US states, continents and major world cities.
  * ``ACRONYMS`` — common tech / IT / maker abbreviations (cpu, gpu, pid, vpn,
    hdmi ...). Two-letter tokens ("it", "ai", "tv") are never judged at all.

topic_hygiene.is_known_word() adds the usual inflections and affixes on top
(plurals, -ed/-ing/-er/-ly/-ness/-ment/-ation, un-/re-/over-, simple
compounds), so "mulching" resolves through "mulch".

It does not need to be exhaustive: an unknown word is only one input to a
"mostly non-words" judgement, and any word the owner has used in two separate
turns, or that two of his stored facts use, is accepted regardless. That is
what lets real client names and jargon he actually uses through.

Stdlib only. No names of people, and nothing that identifies anyone: this
file and the word list ship in the public repo.
"""
from __future__ import annotations

import os
import threading

_WORDS = """
a able about above absolute accept access accident account accurate achieve
acid acre across act action active activity actor actual adapt add address
adjust admin admire admit adopt adult advance advantage adventure advert advice
advise affair affect afford afraid after afternoon afterward again against age
agency agenda agent ago agree ahead aid aim air aircraft airline airport alarm
album alcohol alert alien alike alive all allergy allow almost alone along
aloud already also alter alternative although altogether always amateur amaze
among amount ample amuse analog analyse analysis analyze ancient and angle
angry animal ankle anniversary announce annoy annual another answer ant
anxiety anxious any anybody anyhow anyone anything anyway anywhere apart
apartment apology app apparent appeal appear appetite apple appliance
application apply appoint appointment appreciate approach appropriate approve
approximate apron arcade arch architect architecture area argue argument arise
arm armchair army around arrange array arrest arrival arrive arrow art article
artist artwork as ash aside ask asleep aspect assembly assess asset assign
assist assistant associate assume assure at athlete atmosphere atom attach
attack attempt attend attention attic attitude attract audience audio august
aunt author auto automatic autumn available average avoid awake award aware
away awesome awful awkward axis

baby back backdrop background backpack backup backyard bacon bad badge bag
bake baker balance balcony ball balloon ban banana band bandwidth bank bar
barbecue bare bargain bark barn barrel base baseball basement basic basin
basket basketball bat batch bath bathroom battery battle bay beach bead beam
bean bear beard beat beautiful beauty because become bed bedroom bee beef beer
before beg begin behalf behave behavior behaviour behind being belief believe
bell belong below belt bench bend beneath benefit berry beside best bet better
between beverage beyond bicycle bid big bike bill billion bin bind biology bird
birth birthday biscuit bit bite bitter black blade blame blank blanket blast
blend bless blind block blog blood blow blue board boat body boil bold bolt
bomb bond bone bonus book boot border bore born borrow boss both bother bottle
bottom bounce boundary bow bowl box boy brain brake branch brand brass brave
bread break breakfast breath breathe brick bridge brief bright brilliant bring
broad broadcast brother brown browser brush bubble bucket budget buffer bug
build builder bulb bulk bull bullet bump bunch burger burn burst bury bus
business busy but butter button buy buzz by

cabin cabinet cable cafe cage cake calculate calendar call calm camera camp
campaign campus can canal cancel cancer candidate candle candy cap capable
capacity capital captain capture car carbon card care career careful cargo
carpet carrot carry cart cartoon case cash cast castle casual cat catalog
catch category cause ceiling celebrate cell cellar cent center centre century
cereal ceremony certain certificate chain chair chalk challenge champion
chance change channel chapter character charge charity charm chart chase chat
cheap cheat check cheek cheer cheese chef chemical chemistry chest chicken
chief child chill chimney chin chip chocolate choice choir choose chop chord
chore chorus church cinema circle circuit citizen city civil claim clap class
classic classroom clay clean clear clerk clever click client cliff climate
climb clinic clip clock close closet cloth clothes cloud club clue coach coal
coast coat code coffee coin cold collapse collar colleague collect college
color colour column combine come comedy comfort comic command comment commerce
commit committee common community company compare compete complain complete
complex component compose compound compress computer concept concern concert
conclude concrete condition conduct conference confident confirm conflict
confuse connect connection conscious consider consist console constant
construct consult consume contact contain container content contest context
continue contract contrast contribute control convenient conversation convert
convince cook cookie cool cooperate cope copper copy cord core corn corner
correct cost costume cottage cotton couch cough could council count counter
country county couple courage course court cousin cover cow crack craft crash
crazy cream create creature credit crew crime crisis crisp critic crop cross
crowd crown crucial cruel cruise crush cry crystal cube cucumber cultural
culture cup cupboard cure curious curl current curry curtain curve cushion
custom customer cut cute cycle

dad daily damage damp dance danger dare dark data database date daughter dawn
day dead deadline deal dear death debate debt debug decade decent decide deck
declare decline decorate deep deer default defeat defend define definite
degree delay delete deliberate delicate delicious delight deliver demand demo
dense dentist deny depart department depend deposit depth describe desert
design desire desk dessert destroy detail detect determine develop device
devote diagram dial diamond diary dictionary die diet differ different
difficult dig digital dinner dinosaur dip direct direction dirt dirty disable
disaster disc discount discover discuss disease dish disk display distance
distinct district disturb dive divide do dock doctor document dog doll dollar
domain domestic door dot double doubt dough down download dozen draft drag
dragon drain drama draw drawer dream dress drift drill drink drive driver drop
drown drug drum dry duck due dull dump during dust duty

each eager ear early earn earth ease east easy eat echo economy edge edit
education effect effort egg eight either elbow elder elect electric
electronic element elephant elevator else email embrace emerge emergency
emotion employ empty enable encounter encourage end enemy energy engage engine
engineer enjoy enormous enough ensure enter entertain entire entrance entry
envelope environment episode equal equipment era error escape essay essential
establish estate estimate etc even evening event eventual ever every evidence
evil exact exam examine example excel excellent except exchange excite excuse
exercise exhaust exhibit exist exit expand expect expense expensive experience
experiment expert explain explode explore export expose express extend extent
extra extreme eye

fabric face fact factor factory fade fail faint fair faith fall false fame
familiar family famous fan fancy fantastic far fare farm fashion fast fat fate
father fault favor favour favorite favourite fear feast feature fee feed feel
fellow female fence festival fetch fever few fiber fibre fiction field fierce
fifteen fifty fight figure file fill film filter final finance find fine
finger finish fire firm first fish fit five fix flag flame flash flat flavor
flavour fleet flight float flood floor flour flow flower flu fluid fly focus
fog fold folder folk follow fond food fool foot football for force forecast
forest forever forget forgive fork form formal format former formula fort
fortune forty forum forward fossil foster found foundation four fox frame free
freeze freight frequent fresh fridge friend fright frog from front frost
frozen fruit fuel full fun function fund funny fur furniture further fuse
future

gadget gain galaxy gallery gallon game gap garage garbage garden garlic gas
gate gather gauge gear gel general generate generous gentle genuine geography
gesture get ghost giant gift ginger girl give glad glance glass global glove
glow glue go goal goat god gold golf good goods govern government grab grace
grade grain gram grand grandfather grandmother grant grape graph graphic grass
grateful gravity great green greet grey grid grill grin grind grip grocery
ground group grow growth guarantee guard guess guest guide guilt guitar gun gut
guy gym

habit hair half hall hammer hand handle hang happen happy harbor harbour hard
hardware harm harvest hat hate have hay head headline headphone health hear
heart heat heater heaven heavy hedge heel height hell hello helmet help hen
herb here hero hide high highway hike hill hint hip hire history hit hobby hold
hole holiday hollow holy home homework honest honey hook hope horizon horn
horror horse hospital host hot hotel hour house household housing how however
hub huge human humor humour hundred hunger hunt hurry hurt husband hut

ice icon idea ideal identify identity idle if ignore ill illegal image
imagine immediate impact import important impose impress improve in inch
incident include income increase indeed index indicate individual indoor
industry infant infection influence inform ingredient initial injure injury
ink inner innocent input insect insert inside insight insist inspect inspire
install instance instant instead institute instruction instrument insurance
intend intense interest internal internet interview into introduce invent
invest investigate invitation invite involve iron island issue it item

jacket jam jar jaw jazz jeans jelly jet jewel job jog join joint joke journal
journey joy judge juice jump jungle junior junk jury just justice

keen keep kettle key keyboard kick kid kill kind king kiss kit kitchen kite
knee knife knit knock knot know knowledge

lab label labor labour lace lack ladder lady lake lamb lamp land landscape
lane language lap laptop large laser last late later laugh launch laundry law
lawn lawyer lay layer lazy lead leader leaf league leak lean learn lease least
leather leave lecture left leg legal legend leisure lemon lend length lens
less lesson let letter level lever liberty library licence license lid lie
life lift light like likely limit line link lion lip liquid list listen
literature litter little live load loaf loan local lock log logic lonely long
look loop loose lord lose loss lot loud lounge love low loyal luck lunch lung
luxury

machine mad magazine magic magnet mail main maintain major make male mall man
manage manner manual manufacture many map marathon marble march margin mark
market marriage marry mask mass master match mate material math mathematics
matter maximum may maybe meal mean measure meat mechanic media medical
medicine medium meet meeting melody melt member memory mental mention menu
mercy mere merge mess message metal meter method metre middle midnight might
mild mile milk mill million mind mine mineral minimum minor minute mirror miss
mission mistake mix mobile mode model modern modest module moment money
monitor monkey month mood moon moral more morning most mother motion motor
mountain mouse mouth move movie much mud mug multiple murder muscle museum
mushroom music must mutual mystery myth

nail name narrow nation native natural nature navy near neat necessary neck
need needle negative neighbor neighbour neither nephew nerve nervous nest net
network neutral never new news newspaper next nice niece night nine no noble
nobody nod noise none noon nor normal north nose not note nothing notice novel
now nowhere number nurse nut

oak object observe obtain obvious occasion occur ocean odd of off offer office
officer often oil okay old olive on once one onion online only onto open opera
operate opinion opponent opportunity oppose option or orange orbit order
ordinary organ organise organize origin other otherwise ought ounce our out
outcome outdoor outline output outside oven over overall owe own owner oxygen

pace pack package pad page pain paint pair palace pale palm pan panel panic
pants paper parade paragraph parcel parent park parking part particle
particular partner party pass passage passenger passion password past pasta
paste patch path patience patient pattern pause pay peace peach peak peanut
pear pen pencil penny people pepper per percent perfect perform perhaps period
permanent permit person personal pet phase phone photo photograph phrase
physical physics piano pick picnic picture pie piece pig pile pill pillow
pilot pin pink pipe pitch pity pixel pizza place plain plan plane planet plant
plastic plate platform play player pleasant please pleasure plenty plot plug
plus pocket podcast poem poet point poison pole police policy polish polite
political pollution pond pool poor pop popular population porch pork port
portion portrait pose position positive possess possible post poster pot
potato pound pour powder power practical practice praise pray predict prefer
pregnant prepare present preserve president press pressure pretend pretty
prevent previous price pride priest primary prime prince print printer prior
priority prison private prize probable problem procedure process produce
product profession professor profile profit program programme progress project
promise promote prompt proof proper property proposal propose protect protein
protest proud prove provide public publish pull pump punch pupil puppy
purchase pure purple purpose purse push put puzzle

quality quantity quarter queen query question queue quick quiet quilt quit
quite quiz quote

rabbit race rack radio rage rail rain raise random range rank rapid rare rat
rate rather ratio raw reach react read ready real realise realize reason
recall receipt receive recent recipe recognise recognize recommend record
recover red reduce refer reflect reform refresh refrigerator refund refuse
regard region register regret regular reject relate relative relax release
relevant relief religion rely remain remark remember remind remote remove rent
repair repeat replace reply report represent request require rescue research
reserve resident resist resolve resort resource respect respond rest
restaurant result retail retire return reveal review reward rhythm rib rice
rich rid ride right ring rise risk river road roast rob robot rock role roll
romance roof room root rope rose rough round route routine row royal rub
rubber rubbish rude rug ruin rule run rush

sad safe sail salad salary sale salmon salt same sample sand sandwich satisfy
sauce save saw say scale scan scene schedule scheme school science score
scratch scream screen screw script sea search season seat second secret
section secure see seed seek seem select self sell send senior sense sensor
sentence separate sequence series serious servant serve server service session
set settle seven several severe sew shade shadow shake shall shallow shame
shape share sharp shave she shed sheep sheet shelf shell shelter shield shift
shine ship shirt shock shoe shoot shop shore short should shoulder shout show
shower shut shy sick side sight sign signal silence silk silly silver similar
simple since sing single sink sir sister sit site situation six size skate
sketch ski skill skin skirt sky sleep slice slide slight slim slip slope slot
slow small smart smell smile smoke smooth snack snake snap snow so soap soccer
social society sock soft software soil soldier solid solution solve some
somebody somehow someone something sometime somewhat somewhere son song soon
sore sorry sort soul sound soup sour source south space spare speak special
species specific speech speed spell spend spice spider spill spin spirit split
spoil sponsor spoon sport spot spray spread spring spy square squeeze stable
staff stage stair stake stamp stand standard star start state station stay
steady steak steal steam steel steep step stick still stock stomach stone stop
store storm story stove straight strange stranger strategy straw stream street
strength stress stretch strict strike string strip stroke strong structure
struggle student studio study stuff stupid style subject submit substance
subtle succeed success such suck sudden suffer sugar suggest suit summary
summer sun super supper supplement supply support suppose sure surface surgery
surprise surround survey survive suspect swap sweat sweep sweet swim swing
switch sword symbol sympathy system

table tablet tag tail take tale talent talk tall tank tap tape target task
taste tax taxi tea teach team tear tech technical technique technology teen
teeth telephone television tell temperature temple temporary ten tend tennis
tense tent term terrible test text than thank that the theater theatre theft
their theme then theory therapy there these thick thief thin thing think third
thirsty thirty this those though thought thousand thread threat three throat
through throw thumb thunder thus ticket tidy tie tiger tight tile till timber
time tin tiny tip tire tired tissue title to toast today toe together toilet
tomato tomorrow ton tone tongue tonight too tool tooth top topic torch total
touch tough tour toward towel tower town toy trace track trade tradition
traffic trail train transfer transform transport trap trash travel tray treat
tree trend trial triangle trick trip trophy trouble truck true trust truth try
tube tune tunnel turn tutor twelve twenty twice twin twist two type typical

ugly umbrella uncle under understand uniform union unique unit unite universe
university unless until up update upon upper upset upstairs urban urge urgent
us use useful usual

vacation vacuum valid valley value van vanilla variety various vary vast
vegetable vehicle venture venue verb version very vessel via video view
village vine violin virtual virus visible vision visit visual vital voice
volume volunteer vote

wage wait wake walk wall wallet wander want war warm warn wash waste watch
water wave way we weak wealth weapon wear weather web website wedding week
weekend weigh weight welcome well west wet what whatever wheat wheel when
where whether which while whip whisper whistle white who whole why wide wife
wild will win wind window wine wing winner winter wipe wire wise wish with
within without witness wolf woman wonder wood wool word work worker workshop
world worry worth would wound wrap wrist write wrong

yard yeah year yellow yes yesterday yet yield yogurt you young youth

zero zone zoo

am are is was were been be being has had having does did done doing
went gone ran saw seen gave given took taken came wrote written spoke spoken
broke broken chose chosen drove driven ate eaten fell fallen flew flown forgot
forgotten froze got gotten grew grown hid hidden knew known lay lain rode
ridden rang rung rose risen sang sung sank sunk shook shaken shot shrank slept
slid sold sent spent spun stood stole stolen stuck struck swore sworn swam
swum taught tore torn told thought threw thrown understood woke woken wore worn
won wound built bought brought caught fought found had heard held kept laid
led left lent lost made meant met paid said sat sought slept sped told
children men women people feet teeth mice geese lives knives wives leaves
halves wolves shelves loaves thieves data media criteria phenomena
better best worse worst more most less least further farther
i me my mine myself you your yours yourself yourselves he him his himself she
her hers herself it its itself we us our ours ourselves they them their
theirs themselves one ones anyone everyone someone no nothing everything
everybody whom whose which each every either neither both few many much
several such own other another same

acoustic adapter adhesive algorithm aluminium aluminum amp amplifier analytics
antenna arduino assembly audiobook automation avatar axle backlight bandwidth
barcode baud benchmark binary bios bitrate blender bluetooth bookmark boot
bot bracket breadboard broadband buffer busbar byte cache calibrate camcorder
capacitor carriage cartridge chassis chatbot checkout chipset clamp clipboard
cloud codec coil compiler config configure connector console controller
cookbook coordinate copier cpu crosshair cursor dashboard datasheet debugger
decoder desktop dial diode directory dock dongle driver drone dryer duct
earbud earphone encoder encrypt endpoint enclosure epoxy ethernet extruder
fan filament firewall firmware flashlight flatbed fork framework gaming gateway
gear gigabyte glitch gps gpu graphics grinder handheld hardcover hashtag hdmi
headset heatsink hinge hologram homepage hoodie hotspot hub infrared inbox
inkjet install interface inverter joystick kernel keyboard keychain keypad
kilobyte kiosk laptop lathe layout led lidar linux lithium login logo macro
magnet mainframe malware megabyte memo menu mesh metadata microchip
microcontroller microphone microwave midi milestone mixer modem monitor
motherboard motorcycle mouse multimeter nozzle notebook notification offline
online opcode oscilloscope outlet overlay pager palette parser patch pc pcb
pedal peripheral pixel playlist plugin plywood podcast pointer polymer port
portal potentiometer preset printer processor profile projector prototype
pulley pulse python quadcopter qr queue radar ram raspberry reactor receiver
recorder reel refill relay remote render repo resin resistor resolution
router runtime sander satellite scanner scooter screenshot screwdriver
script sdk selfie semiconductor sensor server servo setup shader shortcut
signal simulator sketch slicer smartphone smartwatch socket solder solenoid
soundbar speaker spool spreadsheet sprocket ssd stack stepper stereo stylus
subwoofer subscription superglue switchboard sync tablet taskbar telescope
template terminal thermostat thumbnail timeline timer toaster toggle toolbar
toolbox touchpad touchscreen tracker trackpad transistor tripod troubleshoot
tuner tutorial tv upload usb username utility vacuum valve vendor voltage
walkie webcam webpage widget wifi wiki wireless wizard workbench workflow
workstation wrench xbox zip zipper

acrylic anvil axe bolt caliper carpentry chisel clamp compressor countersink
dowel drywall file glue grout hacksaw hammer jigsaw joinery level lumber
mallet miter nail plank plaster pliers plumbing primer rafter rivet sandpaper
saw sawdust screw shim spanner stain stud tile trowel varnish vise washer weld
welding woodwork

apple apricot asparagus avocado bagel banana basil bean beet biscuit blueberry
bread broccoli brownie burrito butter cabbage cake candy carrot casserole
cauliflower celery cereal cheese cherry chili chips cinnamon coconut cookie
corn crab cracker croissant cupcake curry dessert donut doughnut dumpling
espresso fajita fries garlic granola grape gravy hamburger hotdog hummus
jerky kale ketchup lasagna latte lemonade lettuce lime lobster macaroni mango
mayonnaise meatball melon muffin mustard noodle oatmeal omelet omelette onion
orange pancake pasta pastry peach pepperoni pickle pie pineapple popcorn
pork pretzel pudding pumpkin radish raisin ramen ravioli salami salsa sausage
shrimp smoothie soda spaghetti spinach steak stew strawberry sushi taco tofu
toast turkey waffle walnut yogurt zucchini brunch lunch dinner supper snack
takeout delivery leftover leftovers

badminton baseball basketball bowling boxing canoe chess climbing cricket
cycling darts fishing frisbee golf gymnastics hiking hockey jogging karate
kayak marathon pickleball poker rugby running sailing skateboard skating skiing
snowboard soccer softball surfing swimming tennis volleyball wrestling yoga
workout fitness gym treadmill

anime ballet band blues cartoon comedy concert documentary drama fantasy
festival film gallery guitar horror jazz karaoke movie museum musical novel
opera orchestra painting photography piano podcast poetry pottery rap reggae
rock romance sculpture series sitcom soundtrack symphony theater thriller
trivia violin drums bass ukulele saxophone trumpet flute cello harp banjo
keyboard synth synthesizer vinyl

cat dog puppy kitten hamster rabbit parrot goldfish turtle horse pony cow pig
sheep goat chicken duck goose owl eagle hawk crow pigeon deer bear wolf fox
squirrel raccoon skunk mouse rat frog snake lizard spider bee ant butterfly
moth mosquito fly beetle worm whale dolphin shark octopus crab lobster seal
penguin lion tiger elephant giraffe zebra monkey gorilla kangaroo koala panda

ache allergy appointment aspirin bandage blister clinic cold cough dentist
diabetes diagnosis doctor dose fever flu headache health hospital illness
injury insomnia medication nurse pharmacy prescription rash sleep sneeze
stomach surgery symptom therapy vaccine vitamin wellness

account bill budget cash checking credit debit debt deposit expense invoice
loan mortgage paycheck payment payroll pension refund rent salary savings tax
tip wallet insurance premium receipt subscription

classroom college course degree diploma essay exam grade homework lecture
lesson major professor quiz report school semester seminar study syllabus
teacher textbook thesis tuition university assignment lab midterm final
presentation project paper research internship graduation scholarship

airplane airport bicycle boat bus car ferry garage gas highway motorcycle
parking railway road scooter ship station subway taxi traffic train tram
truck tire tyre engine mileage hybrid sedan pickup trailer van

attic balcony basement bathroom bedroom carpet ceiling chimney closet
corridor couch curtain desk dishwasher door drawer driveway faucet fence
fireplace floor furnace garage garden gutter hallway heater kitchen lamp
laundry lawn mailbox mattress microwave mirror oven patio pillow porch
refrigerator roof room rug shed shelf shower sink sofa stairs stove toilet
towel wardrobe window yard dresser nightstand bookshelf bookcase cabinet
countertop pantry

january february march april may june july august september october november
december monday tuesday wednesday thursday friday saturday sunday spring
summer autumn fall winter weekend weekday morning afternoon evening night
midnight noon today tonight tomorrow yesterday week month year decade century
hour minute second moment daily weekly monthly yearly annual lately recently
soon later earlier

zero one two three four five six seven eight nine ten eleven twelve thirteen
fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty
sixty seventy eighty ninety hundred thousand million billion first second
third fourth fifth sixth seventh eighth ninth tenth half quarter dozen

red orange yellow green blue purple violet pink brown black white grey gray
silver gold beige navy teal maroon turquoise

afraid alive amazing angry annoying anxious awake bad beautiful big bored
boring brave bright broken busy calm careful cheap clean clever close cold
comfortable cool crazy cute dangerous dark dead deep delicious different
difficult dirty dry dull early easy empty excited exciting expensive fair
famous fancy fast fat favourite fine flat fresh friendly full funny gentle
glad good great happy hard healthy heavy helpful high honest hot huge hungry
important interesting kind large late lazy light little lonely long loose
loud lovely low lucky mad messy modern narrow nasty natural near neat nervous
new nice noisy normal old open perfect plain pleasant polite poor popular
pretty proud quick quiet rare ready real rich right rough round rude sad safe
scary serious sharp short shy sick silly simple slow small smart smooth soft
sorry special strange strict strong stupid sudden sure sweet tall tasty
terrible thick thin tidy tiny tired tough true ugly unusual useful warm weak
weird wet whole wide wild wise wonderful wrong young

actually almost already also always anyway away back certainly clearly
definitely easily especially eventually exactly fairly finally frankly
generally hardly honestly hopefully however instead just kinda likely mainly
maybe merely mostly nearly never nevertheless nonetheless normally obviously
often only perhaps possibly pretty probably quickly quite rarely rather really
seldom seriously simply slightly sometimes somewhat soon still sort
suddenly surely therefore together too totally truly usually very well
anyways alright okay ok hey hi hello bye goodbye thanks please sorry
yeah yep nope nah uh um hmm oh wow huh

beginning brainstorm build challenge collection deadline design draft
experiment goal hobby idea improvement journey launch milestone plan practice
progress prototype puzzle quest renovation repair research restoration review
rework side skill sprint task test upgrade venture

accessory attachment backpack bag battery belt blanket book bottle box brush
bucket button cable calculator candle card chair charger clock coin comb cup
cushion diary envelope fan flag flashlight glasses glove hat helmet jacket
jar kettle key knife ladder lamp lock magazine map marker mat mug napkin
necklace needle notebook pan paper pen pencil phone picture pillow plate
poster pot purse ring rope ruler scarf scissors sheet shirt shoe soap sock
spoon stamp sticker string suitcase sunglasses tape thermometer ticket tissue
toothbrush towel toy tray umbrella vase wallet watch whistle

address alarm announcement answer argument article audio battery belief
birthday blog brand business call career celebration ceremony chore
comment community company contract conversation delivery email emergency
event feedback forecast gift guest holiday invitation job letter list
meeting message mistake news order package party payment photo plan
question recipe reminder reservation schedule shift shopping story
suggestion survey team trip update vacation visit wedding

alien castle dragon ghost hero legend magic monster myth pirate princess
robot spaceship superhero treasure villain wizard zombie spy detective
mystery crime murder case clue suspect witness trial investigation
episode season finale sequel prequel trailer spoiler

climate cloud desert earthquake flood fog forest hill hurricane island jungle
lake lightning moon mountain ocean planet rain rainbow river sand sea sky snow
soil star storm sun sunrise sunset thunder tornado valley volcano wave weather
wind

accomplish acquire adequate adjacent administer adorable affection aggressive
agriculture alongside altitude ambition amend ancestor anchor angel anger
anonymous anticipate antique applause aquarium arena armor armour aroma
arthritis artificial assault asthma astronomy athletic attendance attorney
auction authentic authority autograph avenue bachelor backbone badly bamboo
banner barber barrier batter beacon beast beaver behold beloved bias biography
blaze bleach blink bliss blizzard blond bloom blossom blueprint blunt blur
blush boast boiler bonfire boom boost booth botany boulder bounty bouquet
boutique bracelet braid bravery breeze brew bribe bride brink bronze brook
broom brutal buckle bud buddy buffet bundle bunk burden bureau burglar
cafeteria calcium calf calorie camel canvas canyon capsule caption caravan
carnival carpenter carve cashier casino catastrophe cathedral cattle caution
cave cavity cedar celebrity cement cemetery census ceramic chaos chapel
charcoal cheerful cherish chew chick chronic cider cigar cigarette circus
citrus clarify clash cloak clone closure clumsy cluster clutch cocktail cocoa
coffin cognitive coincidence collision colony comet commute compact companion
compass compassion compensate competent compile compliment comply comprehend
compromise compute conceal concentrate condo cone confess confetti congress
conquer conscience consent conserve conspiracy constitution contemplate
contempt continent contradict convey cooler copyright coral corporate corrupt
cosmetic cosmic counsel countryside coupon courier courtesy courtyard coward
cozy cradle cramp crane crate crater crawl crayon creek creep crib crispy
crocodile crooked crouch crumb crunch crust cuddle cue cuisine culprit
cultivate curb curfew cyber cylinder cynical dairy dam damn dandelion daring
darling dash daydream daylight dealer debris deceive decimal decision dedicate
deduct defect deficit delegate demolish demonstrate dental departure depress
deploy deputy derive descend deserve despair desperate despite destination
detective devil devour dialog dialogue dictate diesel dignity dilemma dim
dine diner dinosaur diploma diplomat disguise dismiss dispute dissolve
distract distress distribute dive divorce dizzy dome donate donkey doom dose
drastic drawing dread drip drizzle drought dumb dungeon durable dusk dynamic
eagle earring earthquake ebook eclipse ecology edible editor efficient elastic
elbow elderly elegant eligible eliminate elite embarrass emblem embassy emerald
empire enchant endless endure enforce enhance enroll enrol enterprise
enthusiasm entity envy epic equation equator erase erode errand erupt
escalate escort estimate ethic evacuate evaluate evaporate exaggerate excess
exclude exclusive execute exotic expire exploit explosion expo exquisite
fabulous facility faction fairy fake falcon fantasy farewell fascinate feather
federal feeble ferry fertile fiddle fierce fig filth finale fireworks fist
flare flaw flee flesh flexible flick flip flock flourish fluffy flush foam foe
foil forbid forge formation fraction fragile fragment franchise fraud freckle
frenzy friction fringe frontier frustrate fulfil fulfill fumble furious fusion
fuss galaxy gamble garment gasp gaze gear gem generation genius genre gentleman
germ gift gigantic giggle glacier glamour glare gleam glide glimpse glitter
gloom glory glossary gnome goose gorgeous gossip gown graceful gradual
graduate graffiti grammar granite grasp grave gravel graze greed greedy grief
grim grocer groom grumpy guardian guideline guilty gulf gully gust gutter
habitat hack haircut hallway halt hamper handsome harmony harness harsh haste
haunt hazard haze heap heir helicopter hemisphere heritage hesitate hierarchy
hijack hinge hippo hitch hive hoax hockey hoist homeless hood hop hormone
hospitality hostage hostile hug hum humble humid hurdle hurl hybrid hydrogen
hygiene hymn hyphen icicle ideology idiot idol ignite illusion illustrate
imitate immense immune implement imply impulse incense incline incredible
indoor inevitable infinite inflate inherit inhale inject inmate inn innovate
inquire insane inspiration instinct insult integrity intellect intelligence
interior interrupt interval intimate intrigue invade invoice irony irrigate
itch ivory jail janitor jealous jeep jersey jewelry jewellery jolly jumbo
junction justify juvenile kayak kidney kilogram kilometer kilometre kin
kingdom kitten knight koala ladle lamb lament landlord landmark lantern lap
latitude lattice lava lavender lawsuit leash ledge legacy legislation lemon
leopard lettuce lever liability liberal lick lifestyle lifetime limb limp
linen liner linger litre liter lizard lobby lodge lofty logo longitude
lottery lotion lullaby lumber lump lunar lure lush lyric lyrics magnificent
maid majesty mammal mane mango mansion mantle maple margin marine marsh mascot
massage massive mattress maze meadow mechanism medal meditate mellow memorial
mentor merchant mermaid merit metaphor meteor microscope migrate mileage
militia mimic miniature minister miracle mischief miser misery mist moan moat
mock modify moist mold molecule monastery monk monologue monopoly monster
monument moose mop mortal mosque motel motivate motive mound mourn mow mower
mule mumble mural muscle mustache mute mutter muzzle naive nanny napkin
narrate navigate negotiate nerd nibble nickname nightmare nimble nomad
nominate nonsense nostalgia notorious nourish novice nuclear nudge nugget
nuisance nun nurture nylon oath obey oblige obscure obsess obstacle offend
omit opaque opera optical optimism orchard orchestra orchid organism ornament
orphan ostrich outfit outrage oval overdue overflow overlook overwhelm owl
pacific paddle padlock pageant pail palm pamphlet panther paradise paradox
parallel paralyze parchment pardon parish parliament parrot partial pastel
pastry pasture patent patio patrol patron pave pavement pavilion peanut pearl
pebble pedestrian peel peer pelican penalty pendant penguin peninsula
perceive perch peril perimeter perish persist persuade pest petal petition
petrol petty pharmacy pheasant philosophy phobia pierce pigeon pilgrim pinch
pioneer pistol pitcher plague plaid plank plaza plea pledge plow plough pluck
plumber plump plunge poke polar poll pony poodle porcelain portable posture
potent pottery pouch poultry prairie prank precious precise predator prelude
premier premium prestige prey primitive privacy privilege probe proceed
proclaim prodigy profound prohibit prominent prone propel prophet prose
prosper prowl prune psychic psychology publicity pudding puddle pulp pulse
punish pupil puppet pursue pyramid quack quaint qualify quarrel quest quiver
rack radiant radiation radius raft rag rally ranch ransom rash raven razor
realm reap rebel recess recite reckless recycle reef referee refuge regime
rehearse reign rein relic remedy render renew renovate rental repent reptile
republic reputation residue resign resume retreat reunion revenge revenue
reverse revolve revolution ribbon riddle rifle rigid rim ripe ripple ritual
rival roar robe robust rocket rodent rookie roster rot rotate rowdy rubble
ruby rudder rumor rumour rural rust rusty sabotage saddle safari saga sage
saint salmon salon salute sanctuary sandal sane sanitary sarcasm satin sauna
savage savor savour scandal scar scarce scatter scent scholar scoop scorch
scorpion scout scrap scribble scroll scrub sculpt seafood seagull sermon
serpent settlement sewer shaggy shaman shark shatter shepherd sheriff shiver
shrewd shriek shrine shrub shrug shuffle sibling siege sieve silhouette sin
sincere siren skeleton skeptic skull skyscraper slab slam slang slate slave
sled sleek sleeve slender slogan sloppy sloth slum slumber smash smear smirk
smug snail snare sneak sniff snore snort snout snug soar sober solar solemn
sonic soothe sophomore sorrow souvenir sovereign spark sparrow spear spectrum
speculate sphere spike spine spiral splash splendid sponge spontaneous spouse
sprawl sprinkle sprint sprout spur squad squash squat squirrel stab stadium
stagger stain stale stall stammer stance staple starch stare startle starve
statue stature steer stem sterile stew stiff stimulate sting stingy stir
stitch stool stoop stork straddle strain strand strap streak stride stroll
stubborn stumble stun sturdy submarine subsidy suburb subway suede suffix
suitcase sulk summit sunburn superb superior supreme surge surplus swamp swan
swarm sway swear sweater swell swift swirl syllable symphony symptom syndrome
syrup tackle tactic tadpole tailor tame tangle tapestry tariff tattoo tavern
tease tedious telegram temper tempt tenant tender terrace terrain terrific
territory terror testament textile texture thaw theft thermal thorn thrill
thrive throne thrust thug tickle tide tier timid tint toad token toll tomb
topple torment tornado torrent tortoise toxic tractor tragedy trait tramp
tranquil transit translate transparent trauma treaty trek tremble trench
tribe tribute trim trio triumph trivial troll troop tropical trot trout truce
tub tuck tug tulip tumble tuna turbine turbulent turf tusk tutor tweak twig
twilight twinkle tycoon typhoon tyrant ultimate unanimous undergo uphold
upright uproar utensil utmost vague valiant vanish vapor vapour vault veil
vein velvet vendor veteran veto vibrant vicinity vicious victim victory vigil
vigorous villa vintage virtue visa vivid vocabulary vocal vogue volcano vomit
voyage vulture waddle wafer wager wagon waist waiter wail wake walrus wand
warden warehouse warrant warrior wasp weary weave wedge weed weep wheelchair
whale whim whine whirl whisker widow wig wilderness wink wisdom witch wither
wobble woe wolf womb wreath wreck wrestle wriggle wrinkle yacht yarn yawn
yearn yell yolk zeal zebra zest zigzag zinc

api git json html css backend frontend repository compiler container cluster
deploy docker virtual database query server client cache token login logout
signup username password admin browser plugin extension terminal command
shell folder directory printer scanner webcam projector headphones speakers
carburetor carburettor mulch compost fertilizer fertiliser
weedkiller pruning trellis greenhouse seedling sprinkler

"""

COMMON_WORDS: frozenset = frozenset(
    w for w in _WORDS.split() if w.isalpha() and w.islower())

# Countries, US states, continents / regions and major world cities. Multi-word
# names are listed word by word ("new", "zealand"): the check is per word.
_PLACES = """
afghanistan albania algeria andorra angola antigua argentina armenia australia
austria azerbaijan bahamas bahrain bangladesh barbados belarus belgium belize
benin bhutan bolivia bosnia herzegovina botswana brazil brunei bulgaria burkina
faso burundi cambodia cameroon canada cape verde chad chile china colombia
comoros congo costa rica croatia cuba cyprus czechia czech denmark djibouti
dominica dominican ecuador egypt salvador equatorial guinea eritrea estonia
eswatini ethiopia fiji finland france gabon gambia georgia germany ghana greece
grenada guatemala guyana haiti honduras hungary iceland india indonesia iran
iraq ireland israel italy ivory jamaica japan jordan kazakhstan kenya kiribati
korea kosovo kuwait kyrgyzstan laos latvia lebanon lesotho liberia libya
liechtenstein lithuania luxembourg madagascar malawi malaysia maldives mali
malta marshall mauritania mauritius mexico micronesia moldova monaco mongolia
montenegro morocco mozambique myanmar burma namibia nauru nepal netherlands
holland zealand nicaragua niger nigeria macedonia norway oman pakistan palau
palestine panama papua paraguay peru philippines poland portugal qatar romania
russia rwanda samoa marino sao tome saudi arabia senegal serbia seychelles
sierra leone singapore slovakia slovenia solomon somalia sudan spain lanka
suriname sweden switzerland syria taiwan tajikistan tanzania thailand timor
togo tonga trinidad tobago tunisia turkey turkmenistan tuvalu uganda ukraine
emirates britain england scotland wales uruguay uzbekistan vanuatu vatican
venezuela vietnam yemen zambia zimbabwe greenland
africa america americas antarctica asia europe oceania arctic atlantic pacific
caribbean mediterranean scandinavia balkans siberia sahara himalaya himalayas
alabama alaska arizona arkansas california colorado connecticut delaware
florida hawaii idaho illinois indiana iowa kansas kentucky louisiana maine
maryland massachusetts michigan minnesota mississippi missouri montana nebraska
nevada hampshire jersey york carolina dakota ohio oklahoma oregon
pennsylvania rhode tennessee texas utah vermont virginia washington wisconsin
wyoming
tokyo delhi shanghai beijing mumbai osaka cairo dhaka karachi istanbul
kolkata manila lagos rio janeiro tianjin kinshasa guangzhou los angeles moscow
shenzhen lahore bangalore paris bogota jakarta chennai lima bangkok seoul
nagoya hyderabad london tehran chicago chengdu nanjing wuhan luanda ahmedabad
kuala lumpur xian hong kong dongguan hangzhou foshan riyadh santiago baghdad
toronto madrid pune houston dallas khartoum berlin rome athens
vienna prague budapest warsaw stockholm oslo copenhagen helsinki dublin
edinburgh glasgow manchester liverpool birmingham amsterdam rotterdam brussels
zurich geneva lisbon barcelona milan naples munich hamburg frankfurt cologne
kyiv kiev minsk bucharest sofia belgrade zagreb reykjavik montreal vancouver
ottawa calgary sydney melbourne brisbane perth adelaide auckland wellington
johannesburg nairobi addis ababa casablanca tunis algiers accra dakar
jerusalem tel aviv dubai abu dhabi doha kabul tashkent almaty hanoi saigon
taipei kyoto yokohama pyongyang ulaanbaatar kathmandu colombo
islamabad havana kingston caracas quito montevideo buenos aires
brasilia recife phoenix philadelphia antonio diego jose austin
jacksonville columbus charlotte indianapolis francisco seattle denver boston
nashville detroit portland vegas louisville baltimore milwaukee albuquerque
tucson fresno sacramento atlanta miami orlando tampa honolulu anchorage
omaha raleigh cleveland pittsburgh cincinnati minneapolis orleans
"""

# Common tech / IT / networking / maker abbreviations, lowercase. Two-letter
# ones are never judged, so only 3+ letters matter here.
_ACRONYMS = """
pid cpu gpu usb api vpn hdmi ssd hdd nvme ram rom bios uefi lan wan wifi dns
dhcp tcp udp http https ssh ftp sftp smb nas raid ups psu led lcd oled rgb
dmx midi mqtt iot llm gpt nlp ocr tts stt asr pdf csv json xml html css sql
php jpeg jpg png gif svg wav flac aac url uri gui cli ide sdk rest soap
crm erp rmm msp saas paas iaas aws gcp azure cad cnc pla abs petg tpu pcb smd
esp gpio uart spi pwm adc dac fpga plc hmi scada vfd rpm mph kph psi gps
rfid nfc sim lte voip sip pbx sms mms otp mfa sso ldap gpo mdm edr xdr siem
soc ids ips waf vlan nat poe isp qos ssid wpa wep ntp snmp icmp arp bgp ospf
ssl tls pki cert csr jwt oauth saml scim kpi roi eta faq fyi asap diy tbd
pto hvac suv atv dvd dvr nvr cctv ptz hdr fps dpi ppi wpm gpm btu kwh mah
amp amps ohm ohms rtc eeprom sram dram ddr smtp imap pop vnc rdp kvm
vram cuda npu tpm yaml toml ini exe dll msi apk ios macos linux unix ubuntu
debian fedora windows android chromebook chrome firefox safari edge github
gitlab npm pip conda docker kubernetes nginx apache sqlite mysql postgres
redis ffmpeg obs vlc zip rar iso img cpp ascii utf unicode emoji
"""

PLACES: frozenset = frozenset(_PLACES.split())
ACRONYMS: frozenset = frozenset(_ACRONYMS.split())

_WORDLIST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "english_words.txt")
_LEXICON = None
_LEXICON_LOCK = threading.Lock()


def _read_wordlist(path: str = _WORDLIST_FILE) -> frozenset:
    """The bundled word list, or an empty set if it is missing or unreadable
    (the hand lists still work: degrade, never raise)."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return frozenset(w for w in (ln.strip() for ln in fh)
                             if w and not w.startswith("#") and w.isalpha())
    except OSError:
        return frozenset()


def lexicon() -> frozenset:
    """Every known word: the bundled list plus the hand lists. Built once, on
    the first call, then cached."""
    global _LEXICON
    lex = _LEXICON
    if lex is not None:
        return lex
    with _LEXICON_LOCK:
        if _LEXICON is None:
            _LEXICON = _read_wordlist() | COMMON_WORDS | PLACES | ACRONYMS
        return _LEXICON
