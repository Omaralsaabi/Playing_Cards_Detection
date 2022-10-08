import random
import keyboard

# Author: Andria Atalla

# 15-112 Final Project

# This is a Tarneeb game

# DO NOT run on "python.exe"



class Card:

    def __init__(self, val, suit):
        self.suit = suit
        self.rank = val

    # Implementing build in methods so that you can print a card object

    def __repr__(self):
        return self.show()

    def show(self):
        return "{} {}".format(self.rank, self.suit)


class Deck:
    def __init__(self):
        self.cards = []
        self.build()

    # Display all cards in the deck
    def show(self):
        for card in self.cards:
            print(card.show())

    # Generate 52 cards
    def build(self):
        self.cards = []
        for suit in ['Hearts', 'Clubs', 'Diamonds', 'Spades']:
            for val in [2,3,4,5,6,7,8,9,10,'Jack','Queen','King','Ace']:
                self.cards.append(Card(val, suit))

    # Shuffle the deck
    def shuffle(self):
        random.shuffle(self.cards)

    # TODO: modify this, get top card by camera
    # Return the top card
    def deal(self):
        return self.cards.pop()


class Player:
    def __init__(self, name):
        self.name = name
        self.hand = []
        self.fakeHand=[]
        self.Clubs=[]
        self.Diamonds=[]
        self.Hearts=[]
        self.Spades=[]
    # Draw n number of cards from a deck
    # Returns true in n cards are drawn, false if less then that

    def draw(self, deck):

        self.Clubs=[]
        self.Diamonds=[]
        self.Hearts=[]
        self.Spades=[]

        # TODO: modify, player gets 13 cards by camera as input
        for i in range(13):
            card = deck.deal()
            self.hand.append(card)

        for i in self.hand:
            if i.suit == "Clubs":
                self.Clubs.append(i)
            elif i.suit == "Diamonds":
                self.Diamonds.append(i)
            elif i.suit == "Hearts":
                self.Hearts.append(i)
            elif i.suit == "Spades":
                self.Spades.append(i)

        ranks=[2, 3, 4, 5, 6, 7, 8, 9, 10, 'Jack', 'Queen', 'King', 'Ace']
        # order cards for every suit to produce self.hand
        self.fakeHand=[self.Clubs,self.Diamonds,self.Hearts,self.Spades]
        for i in self.fakeHand:
            for k in range(len(i)):
                least=i[k]
                for j in range(len(i)):
                    if ranks.index(i[j].rank)<ranks.index(least.rank):
                        least=i[j]
                        i[j]=i[k]
                        i[k]=least
        self.hand=self.Clubs+self.Diamonds+self.Hearts+self.Spades

    def showHand(self):

        print ("{}'s hand: {} \n".format(self.name,self.hand))

    def discard(self):
        return self.hand.pop()

def getSuit(x):
    return x.suit

def distribute(t_names):
    global player_names
    global player1
    global player2
    global player3
    global player4
    global p1_name
    global p2_name
    global p3_name
    global p4_name

    if t_names == None:
        # function distributing cards to players and making the players global variable for later use
        player_names = {}

        myDeck = Deck()
        myDeck.shuffle()

        p1_name=input("Player 1 -> Enter name: ")
        player1 = Player(p1_name)
        player1.draw(myDeck)
        player1.showHand()

        p2_name = input("Player 2 -> Enter name: ")
        player2 = Player(p2_name)
        player2.draw(myDeck)
        player2.showHand()

        p3_name = input("Player 3 -> Enter name: ")
        player3 = Player(p3_name)
        player3.draw(myDeck)
        player3.showHand()

        p4_name = input("Player 4 -> Enter name: ")
        player4 = Player(p4_name)
        player4.draw(myDeck)
        player4.showHand()

        player_names = {'0': p1_name, '1': p2_name, '2': p3_name, '3': p4_name}
    else:
        print('distribute: proceeding with previous player names..')
        player_names = {}

        myDeck = Deck()
        myDeck.shuffle()

        p1_name = t_names['0']
        player1 = Player(p1_name)
        player1.draw(myDeck)
        player1.showHand()

        p2_name = t_names['1']
        player2 = Player(p2_name)
        player2.draw(myDeck)
        player2.showHand()

        p3_name = t_names['2']
        player3 = Player(p3_name)
        player3.draw(myDeck)
        player3.showHand()

        p4_name = t_names['3']
        player4 = Player(p4_name)
        player4.draw(myDeck)
        player4.showHand()
        player_names = {'0': p1_name, '1': p2_name, '2': p3_name, '3': p4_name}
        print(f'Player 1: {p1_name}')
        print(f'Player 2: {p2_name}')
        print(f'Player 3: {p3_name}')
        print(f'Player 4: {p4_name}')
        print()

    return myDeck

def resort(players,x):
    # helper function for resotring players after each round
    for i in players:
        if i.name==x:
            players=players[players.index(i):]+players[:players.index(i)]
    return players

def start(inp, t_data, t_names=False):
    # main functions with bidding and playing for user and AI
    print ("Welcome to my Tarneeb game!\n")
    myDeck = distribute(t_names)
    bids = []
    players = [player1, player2, player3, player4]

    def bid():
        bs = ["7", "8", "9", "10", "11", "12", "13"]

        bids = []
        global trump
        global greatest_player

        for i in range(4):
            p_name = player_names[str(i)]
            correct_input = False
            while not correct_input:
                bidsNumber = input(p_name + "'s bid: ")
                correct_input = False
                if str(bidsNumber) not in bids and str(bidsNumber) in bs:
                    bids.append(bidsNumber)
                    correct_input = True
                elif bidsNumber == '13':
                    bids.append("13")
                    correct_input = True
                elif bidsNumber == "pass":
                    bids.append(bidsNumber)
                    correct_input = True
                else:
                    try:
                        int(bidsNumber)
                        if int(bidsNumber) >= 7 and str(bidsNumber) not in bs:
                            print('Input should be higher than the bid before, or you should pass, try again\n')
                        else:
                            print(f'wrong range of number should be one of these values {bs}, try again\n')
                    except:
                        print("Input should be an integer from 7 to 13 or 'pass', try again\n")

                if bids.count("pass") == 4:
                    print(f"Player 4 ({p_name}) must bid if everybody else is passing")
                    correct_input = False

            if bids[i] != "pass":
                bs = bs[bs.index(bids[i]):]
            print("Player " + p_name + "'s bid: " + str(bids[i]))

        greatest = 0
        greatest_player = "unknown"
        for i, bid_i in enumerate(bids):
            try:
                bid_i = int(bid_i)
                if type(bid_i) == int:
                    if bid_i > greatest:
                        greatest = bid_i
                        greatest_player = player_names[str(bids.index(str(bid_i)))]
                bids[i] = int(bids[i])
            except:
                bids[i] = -1

        # who is greatest bidder?
        print(f'greatest bidder is {greatest_player} (bid: {greatest})')

        global greatestBidder
        # choosing trump
        for p in players:
            if p.name == greatest_player:
                greatestBidder = p
                print("\nHey " + greatestBidder.name + " , please select Spades, Diamonds, Hearts, or Clubs")
                trump = input("Select Suit:")
                while trump not in ["Spades", "Diamonds", "Hearts", "Clubs"]:
                    print("Wrong suit! choose Spades,Diamonds,Hearts,Clubs")
                    trump = input("Select Suit:")
                print(f"the trump is {trump}, let's start the round")
        return bids

    # bid function
    bids = bid()

    # preparing data
    data = t_data.copy()
    data['trump'] = trump
    data['greatestBidder'] = greatestBidder
    data['points'] = {}
    data['players'] = players
    data['discardedCards'] = []
    data['playedCards'] = []
    data['teamAssignments'] = {'team 1': [player1, player3], 'team 2': [player2, player4]}
    # data['teamBids'] = {'team 1': max([bids[0], bids[2]]), 'team 2': max([bids[1], bids[3]])}
    data['teamWins'] = {'team 1': 0, 'team 2': 0}

    if greatestBidder in data['teamAssignments']['team 1']:
        data['bid'] = {'team 1': max(bids), 'team 2': -1}
        print(f"bid to match: {data['bid']}")
    elif greatestBidder in data['teamAssignments']['team 2']:
        data['bid'] = {'team 2': max(bids), 'team 1': -1}
        print(f"bid to match: {data['bid']}")
    else:
        print('max bid set is not for any of the players in either team!')

    return data, True

def play(data_used, inp, c_order, t_rounds):
    print('\n\n')
    print(inp)
    print(data_used)
    data_used_before = data_used.copy()
    # data_used_before['trump'] = data_used['trump']
    # data_used_before['greatestBidder'] = data_used['greatestBidder'].copy()
    data_used_before['points'] = data_used['points'].copy()
    data_used_before['players'] = data_used['players'].copy()
    data_used_before['discardedCards'] = data_used['discardedCards'].copy()
    data_used_before['playedCards'] = data_used['playedCards'].copy()

    print(f"Tarneeb ({data_used['trump']}): round {t_rounds + 1}\n")
    print(f"PLAYERS: ")
    for p in data_used['players']:
        print(f'{p} \t {p.name}')
    print()
    print('INPUT\t', inp, end='\n\n')

    # checking if cards input are 4
    card_num = len(inp)
    print(f'Number of Cards detected: {card_num}')
    if card_num != 4:
        print("number of inputs isn't correct! must be 4 cards, restart round please")
        return data_used_before, t_rounds
    else:
        print('four cards detected, proceeding...')
        print()

    # assign player 1,2,3,4 based on position
    bottom = None
    right = None
    top = None
    left = None
    min_y = 10000
    # max_y = 0
    min_x = 10000
    max_x = 0
    for x, y in inp.values():
        if x < min_x:
            min_x = x
        if x > max_x:
            max_x = x
        if y < min_y:
            min_y = y

    # conversion of card types
    dict_conversion = {'2c': '2 Clubs', '2d': '2 Diamonds', '2h': '2 Hearts', '2s': '2 Spades',
                       '3c': '3 Clubs', '3d': '3 Diamonds', '3h': '3 Hearts', '3s': '3 Spades',
                       '4c': '4 Clubs', '4d': '4 Diamonds', '4h': '4 Hearts', '4s': '4 Spades',
                       '5c': '5 Clubs', '5d': '5 Diamonds', '5h': '5 Hearts', '5s': '5 Spades',
                       '6c': '6 Clubs', '6d': '6 Diamonds', '6h': '6 Hearts', '6s': '6 Spades',
                       '7c': '7 Clubs', '7d': '7 Diamonds', '7h': '7 Hearts', '7s': '7 Spades',
                       '8c': '8 Clubs', '8d': '8 Diamonds', '8h': '8 Hearts', '8s': '8 Spades',
                       '9c': '9 Clubs', '9d': '9 Diamonds', '9h': '9 Hearts', '9s': '9 Spades',
                       '10c': '10 Clubs', '10d': '10 Diamonds', '10h': '10 Hearts', '10s': '10 Spades',
                       'Jc': 'Jack Clubs', 'Jd': 'Jack Diamonds', 'Jh': 'Jack Hearts', 'Js': 'Jack Spades',
                       'Qc': 'Queen Clubs', 'Qd': 'Queen Diamonds', 'Qh': 'Queen Hearts', 'Qs': 'Queen Spades',
                       'Kc': 'King Clubs', 'Kd': 'King Diamonds', 'Kh': 'King Hearts', 'Ks': 'King Spades',
                       'Ac': 'Ace Clubs', 'Ad': 'Ace Diamonds', 'Ah': 'Ace Hearts', 'As': 'Ace Spades'
    }
    placement_order = []
    for card, (x, y) in inp.items():
        if x == min_x:
            left = [card, (x, y)]
            left[0] = dict_conversion[left[0]]
            placement_order.append(left)
        elif x == max_x:
            right = [card, (x, y)]
            right[0] = dict_conversion[right[0]]
            placement_order.append(right)
        elif y == min_y:
            top = [card, (x, y)]
            top[0] = dict_conversion[top[0]]
            placement_order.append(top)
        else:
            bottom = [card, (x, y)]
            bottom[0] = dict_conversion[bottom[0]]
            placement_order.append(bottom)
    print(f"PLACEMENTS --> p1:{bottom}\n p2:{right}\n p3:{top}\n p4:{left}")
    print(f"old order {placement_order}")
    cut_off = data_used['players'].index(data_used['greatestBidder'])
    data_used['players'] = data_used['players'][cut_off:] + data_used['players'][:cut_off]
    print(f"new order {placement_order}")
    print()

    # placement order check with Player objects
    print(f"if {placement_order[0][0]} in {list(map(lambda x: f'{x.rank} {x.suit}', data_used['players'][0].hand))}")
    if placement_order[0][0] in list(map(lambda x: f"{x.rank} {x.suit}", data_used['players'][0].hand)):
        print('Order of cards placed is correct! proceeding')
    else:
        print('Cards Placement order is incorrect,')
        print(f"restart the round with new input please")
        return data_used_before, t_rounds
    print()

    # (play card + order card) check
    for i in range(4):
        # confirmed whether each player played a card in their hand
        c = placement_order[i][0]
        found = False
        player_i = data_used['players'][i]
        print(f"checking {player_i.name}'s Cards: {player_i.hand}")
        print(f"{player_i.name} played{c}")
        if c in list(map(lambda x: f"{x.rank} {x.suit}", player_i.hand)):
            found = True
        if found:
            print(f"player {i+1} ({player_i.name}) ---> confirmed his card is {c}")
        else:
            print(f"player {i+1} ({player_i.name}) ---> does not have the card {c}")
            print(f"restart the round with new input please")
            return data_used_before, t_rounds
        print()


        c_order[i] = dict_conversion[c_order[i].split(' ')[0]]
        # order of card check
        print(f"{c}(placed) == {c_order[i]}(input order {i+1})")
        if c == c_order[i]:
            print(f"player {i+1} ({player_i.name}) ---> input order is consistent with placement order")
        else:
            print(f"player {i+1} ({player_i.name}) ---> input order is inconsistent with placement order!")
            print(f"restart the round with new input please")
            return data_used_before, t_rounds
        print()
    print()

    print('all good')
    while not keyboard.is_pressed('r'):
        pass

    # TODO: use
    # data_used['players']
    # c_order
    # data_used['Tricks']
    # data_used['playedCards']
    # data_used['trump']
    # data_used['discardedCards']
    print(f"\n\nPHASE 2 (round {t_rounds})")
    for p_c in data_used['players']:
        print(f"{p_c.name}'s cards for the current round: {p_c.hand}")
    print(c_order)

    # adding played cards
    for i in range(4):
        if c_order[i] not in data_used['discardedCards']:
            data_used['playedCards'] += [c_order[i]]
        else:
            print(f"{data_used['players'][i]} played this card!")
    print(f"used cards for the round {data_used['playedCards']}")
    print('\n')

    # Apply tarneeb rules for the round
    winner = None
    trump_exists = False
    r_convresions = {-1:-1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, '10':10 ,'Jack': 11, 'Queen': 12, 'King': 13, 'Ace': 14}
    for i, c_played in enumerate(data_used['playedCards']):
        print(f"-------{i+1}: {c_played} in {data_used['players'][i].hand}-------", end='\n')
        print(f'did player play the card {c_played}', end=': ')
        if c_played in list(map(lambda x: f"{x.rank} {x.suit}", data_used['players'][i].hand)):
            print('no')
            if i == 0:
                kind_r, kind_s = c_played.split(' ')
                kind_r = r_convresions[kind_r]
                if kind_s == data_used['trump']:
                    trump_r, trump_s = c_played.split(' ')
                else:
                    trump_r, trump_s = -1, data_used['trump']
                trump_r = r_convresions[trump_r]

            # determine winner of round
            r, s = c_played.split(' ')
            r = r_convresions[r]
            player_suits = list(map(lambda x: f"{x.suit}", data_used['players'][i].hand))
            if ((s == kind_s) and (r >= kind_r) and (not trump_exists)):
                winner = data_used['players'][i]
                kind_r = r
            elif ((s == kind_s)):
                pass
            elif (kind_s in player_suits):
                print(f"since the first player ({data_used['players'][0].name}) played a card of type {kind_s},\
{data_used['players'][i].name} must play a card of that type too, since he has it")
                print(f"round canceled, restart the round with new input please")
                return data_used_before, t_rounds

            if (s == trump_s) and (r > trump_r):
                winner = data_used['players'][i]
                trump_r = r
                trump_exists = True
        else:
            print('yes')
            print(f"restart the round with new input please")
            return data_used_before, t_rounds

    for i, c_played in enumerate(data_used['playedCards']):
        if c_played in list(map(lambda x: f"{x.rank} {x.suit}", data_used['players'][i].hand)):
            r, s = c_played.split(' ')
            r = r_convresions[r]

            # removing played cards for round from hand
            k = 0
            length = len(data_used['players'][i].hand)
            while k < length:
                r_removal = list(r_convresions.keys())[list(r_convresions.values()).index(r)]
                # print(f"{r} --> {r_removal}")
                if r_removal == str(data_used['players'][i].hand[k].rank) and s == data_used['players'][i].hand[k].suit:
                    c_removed = data_used['players'][i].hand.pop(k)
                    print(f"REMOVED {c_removed} from the cards = {data_used['players'][i].hand}")
                    k -= 1
                    length -= 1
                else:
                    k += 1
    print('-' * 50)



    # finalizing the winner of round
    if winner.name not in data_used['points']:
        data_used['points'][f'{winner.name}'] = 1
    else:
        data_used['points'][f'{winner.name}'] += 1
    data_used['greatestBidder'] = winner
    data_used['discardedCards'].extend(data_used['playedCards'])
    data_used['playedCards'] = []

    print('----------------------- ROUND RESULT -----------------------')
    print(f'winner: {winner.name}')
    print('Team Points')
    for team_name in data_used['teamAssignments'].keys():
        team_players = data_used['teamAssignments'][team_name]
        if winner in team_players:
            data_used['teamWins'][team_name] += 1
        print(f"{team_name}: {data_used['teamWins'][team_name]} Points")
    print(f"next round is started by: {data_used['greatestBidder'].name}")
    for i, p_c in enumerate(data_used['players']):
        print(f"{p_c.name}'s cards for the next round: {p_c.hand}")
    print(f"Tarneeb is {data_used['trump']}")
    print(f"Discarded Cards: {data_used['discardedCards']}")
    print(f"Player Points: {data_used['points']}")
    print('------------------------------------------------------------')
    print('\ninput cards for the next round please')

    return data_used, t_rounds + 1


