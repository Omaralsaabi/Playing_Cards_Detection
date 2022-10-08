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

def distribute():
    # function distributing cards to players and making the players global variable for later use
    global player_names
    player_names = {}

    myDeck = Deck()
    myDeck.shuffle()
    global player1
    global p1_name

    p1_name=input("Player 1 -> Enter name: ")
    player1 = Player(p1_name)
    player1.draw(myDeck)
    player1.showHand()

    global player2
    global p2_name
    p2_name = input("Player 2 -> Enter name: ")
    player2 = Player(p2_name)
    player2.draw(myDeck)
    player2.showHand()

    global player3
    global p3_name
    p3_name = input("Player 3 -> Enter name: ")
    player3 = Player(p3_name)
    player3.draw(myDeck)
    player3.showHand()

    global player4
    global p4_name
    p4_name = input("Player 4 -> Enter name: ")
    player4 = Player(p4_name)
    player4.draw(myDeck)
    player4.showHand()

    player_names = {'0': p1_name, '1': p2_name, '2': p3_name, '3': p4_name}
    return myDeck

def resort(players,x):
    # helper function for resotring players after each round
    for i in players:
        if i.name==x:
            players=players[players.index(i):]+players[:players.index(i)]
    return players

def start(inp):

    # main functions with bidding and playing for user and AI
    # print('start', inp)

    print ("\nWelcome to my Tarneeb game!!")
    print ("\n")
    myDeck = distribute()
    bids = []
    players = [player1, player2, player3, player4]

    def bid():
        # TODO: players choose their (input not camera)
        bs = ["7", "8", "9", "10", "11", "12", "13"]
        bids = []
        global trump
        global greatest_player

        for i in range(4):
            p_name = player_names[str(i)]
            bidsNumber = input(p_name + "'s bid: ")

            if str(bidsNumber) not in bids and str(bidsNumber) in bs:
                bids.append(bidsNumber)
            elif bidsNumber == '13':
                bids.append("13")
            elif bidsNumber == "pass":
                bids.append(bidsNumber)
            else:
                try:
                    int(bidsNumber)
                    if int(bidsNumber) >= 7 and str(bidsNumber) not in bs:
                        print('Input should be higher than the bid before, or you should pass, try again\n\n')
                        return None
                    else:
                        print(f'wrong range of number should be one of these values {bs}, try again\n\n')
                        return None
                except:
                    print("Input should be an integer from 7 to 13 or 'pass', try again\n\n")
                    return None

            print('bids layed: ', bids)
            # print('bids layed: ', end='')
            # for i in range(4):
            #     print(f"{player_names[str(i)]}->{bids[i]}", end=', ')
            if bids[i] != "pass":
                bs = bs[bs.index(bids[i]):]

            print("Player " + p_name + "'s bid: " + str(bids[i]))

        greatest = 0
        greatest_player = "unknown"
        for bid_i in bids:
            try:
                bid_i = int(bid_i)
                if type(bid_i) == int:
                    if bid_i > greatest:
                        greatest = bid_i
                        greatest_player = player_names[str(bids.index(str(bid_i)))]
            except:
                continue

        if bids.count("pass") == 4:
            print("error! someone has to bid, the game can't proceed, try again")
            return None

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
        return True

    # TODO: use this later
    x1 = bid()
    if x1 == None:
        return None, False

    # TODO: save info in files
    # who starts first: (greateset_bidder or winner_last_round)
    # trump card
    # round number
    # Player cards self.hand
    # trick count for each player
    # discarded cards


    ''' comments hidden '''
    # #AI for bidding
    #
    # bs=["7","8","9","10","11","12","13"]
    #
    # if bids[0]!="pass":
    #     bs=bs[bs.index(bids[0]):]
    #
    # trump=""
    # trumps=[]
    # bidsNumber=0
    #
    # for p in [player2,player3,player4]:
    #
    #     change=False
    #     bidsNumber=0
    #
    #     for s in p.fakeHand:
    #         if len(s) > bidsNumber:
    #             bidsNumber = len(s)
    #             big=s
    #             change=True
    #
    #             if p.fakeHand.index(s)==0:
    #                 trump="Clubs"
    #             elif p.fakeHand.index(s)== 1:
    #                 trump="Diamonds"
    #             elif p.fakeHand.index(s) == 2:
    #                 trump = "Hearts"
    #             elif p.fakeHand.index(s) == 3:
    #                 trump = "Spades"
    #
    #     trumps.append(trump)
    #     p.fakeHand.pop(p.fakeHand.index(big))
    #
    #     for cs in p.fakeHand:
    #         for c in range(len(cs)):
    #             if cs[c].rank in ["Ace","King","Queen"]:
    #                 bidsNumber+=1
    #
    #     if str(bidsNumber) not in bids and str(bidsNumber) in bs:
    #         bids.append(str(bidsNumber))
    #         bs=bs[bs.index(str(bidsNumber)):]
    #     elif bidsNumber>13:
    #         bids.append("13")
    #     else:
    #         bids.append("pass")
    #
    # trumps=[""]+trumps
    #
    # for i in range(1,4):
    #     print ("Player " + str(i + 1)+ " bid: " + str(bids[i]))
    #
    # # go through bids (greatest bid wins)
    # for b in range(len(bids)):
    #     try:
    #         int(bids[b])
    #         bids[b]=int(bids[b])
    #     except:
    #         pass
    #
    # greatest=0
    # for bid_i in bids:
    #     if type(bid_i)==int:
    #         if bid_i > greatest:
    #             greatest = bid_i
    #             trump = input('player i, choose your trump: ')
    #
    # if bids.count("pass")==4:
    #     print (random.choice(["Player 2","Player 3"])+ " says: You can't do that Player 4, you gotta bid")
    #     bids[-1]=7
    #     greatest=7
    #     trump=trumps[-1]
    #     print ("Ok fine, we will play for a 7")
    #
    # # who is greatest bidder?
    # if greatest==bids[0]:
    #     greatestBidder=p1_name
    # else:
    #     greatestBidder="Player "+str(bids.index(greatest)+1)
    #
    # for p in players:
    #     if p.name==greatestBidder:
    #         if p.name==p1_name:
    #             print ("\nHey "+greatestBidder+" , please select Spades, Diamonds, Hearts, or Clubs")
    #             trump=input("Select Suit:")
    #             while trump not in ["Spades", "Diamonds", "Hearts", "Clubs"]:
    #                 print ("Wrong suit! choose Spades,Diamonds,Hearts,Clubs")
    #                 trump=input("Select Suit:")
    #             print ("\n"+random.choice(["Player 2","Player 3","Player4"])+" says: let's beat this tarneeb \n")
    #         else:
    #             print ("\n"+greatestBidder+" , please select Spades, Diamonds, Hearts, or Clubs")
    #             print ("Uhhhh, I choose: "+trump+"\n")
    #             print (random.choice(["Player 2","Player 3","Player 4"])+" says: let's beat this tarneeb \n")

    data = {}
    data['trump'] = trump
    data['greatestBidder'] = greatestBidder
    data['points'] = {}
    data['players'] = players
    data['discardedCards'] = []
    data['playedCards'] = []

    return data, True

def play(data_used, inp, c_order, t_rounds):
    print('\n\n')
    print(inp)
    print(data_used)
    data_used_before = data_used.copy()

    print(f"Tarneeb ({data_used['trump']}): round {t_rounds}\n")
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
    print(f"p1:{bottom}\n p2:{right}\n p3:{top}\n p4:{left}")
    print(f"old order {placement_order}")

    # reorder data_used['players']
    # placement_order should be used to check if placement is correct with reordering
    cut_off = data_used['players'].index(data_used['greatestBidder'])
    data_used['players'] = data_used['players'][cut_off:] + data_used['players'][:cut_off]
    print(f"new order {placement_order}")

    # placement order check with Player objects
    if placement_order[0][0] in list(map(lambda x: f"{x.rank} {x.suit}", data_used['players'][0].hand)):
        print('Order of cards placed is correct! proceeding')
    else:
        print('Cards Placement order is incorrect,')
        print(f"restart the round with new input please")
        while not keyboard.is_pressed('r'):
            pass
        return

    # (play card + order card) check
    print(f'new order: {placement_order}')
    for i in range(4):
        # confirmed whether each player played a card in their hand
        c = placement_order[i][0]
        found = False
        player_i = data_used['players'][i]
        print(f"{player_i.name}'s Cards: {player_i.hand}")
        print(f"new order i {c}")
        if c in list(map(lambda x: f"{x.rank} {x.suit}", player_i.hand)):
            found = True
        if found:
            print(f"player {i+1} ({player_i.name}) ---> confirmed his card is {c}")
        else:
            print(f"player {i+1} ({player_i.name}) ---> does not have the card {c}")
            print(f"restart the round with new input please")
            while not keyboard.is_pressed('r'):
                pass
            return
            return data_used, t_rounds


        c_order[i] = dict_conversion[c_order[i].split(' ')[0]]
        # order of card check
        print(f"{c} == {c_order[i]}")
        if c == c_order[i]:
            print(f"player {i+1} ({player_i.name}) ---> input order is consistent with placement order")
        else:
            print(f"player {i+1} ({player_i.name}) ---> input order is inconsistent with placement order!")
            print(f"restart the round with new input please")
            while not keyboard.is_pressed('r'):
                pass
            return
            return data_used, t_rounds

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
    print(f"\n\n\n PHASE 2 (round {t_rounds})")
    for p_c in data_used['players']:
        print(f"{p_c.name}'s cards for the current round: {p_c.hand}")
    print(c_order)

    # discardedCards = [] # TODO: ??

    for i in range(4):
        if c_order[i] not in data_used['playedCards']:
            data_used['playedCards'] += [c_order[i]]
            print(f"used cards for the round {data_used['playedCards']}")
        else:
            print(f"{data_used['players'][i]} played this card!")

    print('\n')
    # winner = Player Object
    winner = None
    trump_exists = False
    r_convresions = {-1:-1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, '10':10 ,'Jack': 11, 'Queen': 12, 'King': 13, 'Ace': 14}

    for i, c_played in enumerate(data_used['playedCards']):
        print(f"{c_played} in {data_used['players'][i].hand}", end='\n\n')
        if c_played in list(map(lambda x: f"{x.rank} {x.suit}", data_used['players'][i].hand)):

            if i == 0:
                kind_r, kind_s = c_played.split(' ')
                kind_r = r_convresions[kind_r]
                if kind_s == data_used['trump']:
                    trump_r, trump_s = c_played.split(' ')
                else:
                    trump_r, trump_s = -1, data_used['trump']
                trump_r = r_convresions[trump_r]

            r, s = c_played.split(' ')
            r = r_convresions[r]
            if (s == trump_s) and (r > trump_r):
                winner = data_used['players'][i]
                trump_r = r
                # data_used['trump'] = f'{r} {s}'
                trump_exists = True
            elif ((s == kind_s) and (r >= kind_r) and (not trump_exists)):
                winner = data_used['players'][i]
                kind_r = r
                # data_used['trump'] = f'{r} {s}'

            # print(f"removing {c_played} from {data_used['players'][i].hand}")

            k=0
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

        else:
            print(f"the card {c_played} has already been played!")
            return

    # finalizing the winner of round
    if winner.name not in data_used['points']:
        data_used['points'][f'{winner.name}'] = 1
    else:
        data_used['points'][f'{winner.name}'] += 1

    data_used['greatestBidder'] = winner
    print(f'winner: {winner.name}')
    print(f"next round is started by: {data_used['greatestBidder'].name}")
    for i, p_c in enumerate(data_used['players']):
        print(f"{p_c.name}'s cards for the next round: {p_c.hand}")
    print(f"Player Points: {data_used['points']}")

    return data_used, t_rounds + 1








    # TODO: rounds (here input for each round from camera)
    for i in range(13):
        print ("\n")
        # player1.showHand()
        playedCards=[]
        winner=""
        if playedCards == []:
            card = False
            length = len(j.hand)

            while not card:
                k = 0
                # TODO: play card here
                played = input(p1_name + " plays: ")
                played = played.split(" ")

                while len(j.hand) == length and k < length:
                    if played[0] == str(j.hand[k].rank) and played[1] == j.hand[k].suit:
                        p = j.hand.pop(k)
                        playedCards.append(p)
                        card = True
                    k += 1

                if card == False:
                    print(random.choice(
                        ["Player 2", "Player 4"]) + " says: Please stop cheating and play a valid card smh")


        # TODO: player turn
        for j in players:
            if j.name==p1_name:
                if playedCards==[]:
                    card=False
                    length=len(j.hand)

                    while not card:
                        k=0
                        # TODO: play card here
                        played=input(p1_name + " plays: ")
                        played=played.split(" ")

                        while len(j.hand)==length and k<length:
                            if played[0]==str(j.hand[k].rank) and played[1]==j.hand[k].suit:
                                p=j.hand.pop(k)
                                playedCards.append(p)
                                card=True
                            k+=1

                        if card==False:
                            print (random.choice(["Player 2","Player 4"])+" says: Please stop cheating and play a valid card smh")

                # TODO: continue here
                elif playedCards!=[]:
                    su=playedCards[0].suit
                    lst=map(getSuit,j.hand)
                    card=False
                    if su not in lst:

                        card=False
                        length=len(j.hand)

                        while not card:

                            k=0
                            played=(input(p1_name + " plays: "))
                            played=played.split(" ")

                            while len(j.hand)==length and k<length:

                                if played[0]==str(j.hand[k].rank) and played[1]==j.hand[k].suit:

                                    p=j.hand.pop(k)
                                    playedCards.append(p)
                                    card=True

                                k+=1

                            if card==False:

                                print (random.choice(["Player 2","Player 4"])+" says: Please stop cheating and play a valid card smh")

                    else:

                        card=False
                        length=len(j.hand)

                        while not card:

                            k=0
                            played=input(p1_name + " plays: ")
                            played=played.split(" ")

                            while len(j.hand)==length and k<length:

                                if played[0]==str(j.hand[k].rank) and played[1]==j.hand[k].suit and played[1]==su:

                                    p=j.hand.pop(k)
                                    playedCards.append(p)
                                    card=True

                                k+=1

                            if card==False:

                                print (random.choice(["Player 2","Player 4"])+" says: Please stop cheating and play a valid card smh.")

            else:

                if playedCards==[]:

                    temp=filter(lambda x:x.suit!=trump,j.hand)

                    if temp!=[]:

                        g=temp[-1]
                        ranks=[2,3,4,5,6,7,8,9,10,'Jack','Queen','King','Ace']

                        for i in temp:

                            if ranks.index(i.rank)>ranks.index(g.rank) and \
                                    True not in [u.suit==i.suit and ranks.index(u.rank)>ranks.index(i.rank) for u in discardedCards]:
                                g=i

                        if g==temp[-1]:

                            y=temp[-1]
                            if [x for x in temp if ranks.index(x.rank)<ranks.index(y.rank)]:
                                g=[x for x in temp if ranks.index(x.rank)<ranks.index(y.rank)][0]

                        playedCards.append(g)
                        p=j.hand.pop(j.hand.index(g))

                    else:

                        temp=filter(lambda x:x.suit==trump,j.hand)
                        g=temp[-1]
                        ranks=[2,3,4,5,6,7,8,9,10,'Jack','Queen','King','Ace']

                        for i in temp:

                            if ranks.index(i.rank)>ranks.index(g.rank) and \
                                    True not in [u.suit==i.suit and ranks.index(u.rank)>ranks.index(temp[-1].rank) for u in discardedCards]:
                                g=i
                        playedCards.append(g)
                        p=j.hand.pop(j.hand.index(g))

                else:
                    temp=filter(lambda x:x.suit==playedCards[0].suit,j.hand)
                    if temp==[]:
                        test=[x.suit==trump for x in playedCards]
                        temp1=filter(lambda x:x.suit==trump,j.hand)
                        # use other algorithm here with change
                        gr=None
                        temp1.reverse()

                        for i in temp:
                            test=[x>i.rank for x in playedCards]
                            if False in test:
                                gr=i
                                break

                        if gr!=None and temp!=[]:

                            playedCards.append(gr)
                            p=j.hand.pop(j.hand.index(gr))

                        elif gr==None and temp!=[]:

                            temp.reverse()
                            playedCards.append(temp[-1])
                            p=j.hand.pop(j.hand.index(temp[-1]))

                        else:

                            playedCards.append(j.hand[-1])
                            p=j.hand.pop(-1)

                    else:

                        ranks=[2,3,4,5,6,7,8,9,10,'Jack','Queen','King','Ace']
                        gr=temp[-1]
                        test=[x.suit==trump for x in playedCards]

                        if True in test:

                            playedCards.append(temp[-1])
                            p=j.hand.pop(j.hand.index(temp[-1]))

                        else:

                            gr=None
                            temp.reverse()
                            for i in temp:
                                test=[ranks.index(x.rank)>ranks.index(i.rank) for x in playedCards]
                                if False in test:
                                    gr=i
                                    break

                            if gr!=None and temp!=[]:

                                playedCards.append(gr)
                                p=j.hand.pop(j.hand.index(gr))

                            elif gr==None and temp!=[]:

                                temp.reverse()
                                playedCards.append(temp[-1])
                                p=j.hand.pop(j.hand.index(temp[-1]))

                            else:

                                playedCards.append(j.hand[-1])
                                p=j.hand.pop(-1)

                print (j.name+" plays: "+str(p.rank)+" "+p.suit)

        plays=playedCards

        if True in [x.suit==trump for x in plays] and [x.suit==trump for x in plays].count(True)>1:

            ranks
            plays=[x for x in plays if x.suit==trump]
            t=plays[0]
            for i in plays:
                if ranks.index(i.rank)>ranks.index(t.rank):
                    t=i
            t=playedCards.index(t)
            winner=players[t]
            players=resort(players,winner.name)

            if winner.name==p1_name or winner.name== "Player 3":
                yourTricks+=1

            else:
                opponentTricks+=1


        elif True in [x.suit==trump for x in plays] and [x.suit==trump for x in plays].count(True)==1:

            plays=[x for x in plays if x.suit==trump]
            t=plays[0]
            t=playedCards.index(t)
            winner=players[t]
            players=resort(players,winner.name)

            if winner.name==p1_name or winner.name== "Player 3":
                yourTricks+=1

            else:
                opponentTricks+=1

        elif True not in [x.suit==trump for x in plays]:

            plays=[x for x in plays if x.suit==playedCards[0].suit]
            t=plays[0]
            for i in plays:
                if ranks.index(i.rank)>ranks.index(t.rank):
                    t=i
            t=playedCards.index(t)
            winner=players[t]
            players=resort(players,winner.name)

            if winner.name==p1_name or winner.name== "Player 3":

                yourTricks+=1

            else:

                opponentTricks+=1

        print ("\nWinner is: " + winner.name + "\n")
        print ("Your team's tricks: " + str(yourTricks))
        print ("Opponents' tricks: " + str(opponentTricks))
        if greatestBidder==p1_name or greatestBidder== "Player 3":

            if yourTricks==greatest:

                print ("Congratulations, you beat this tarneeb")
                exit()

            elif opponentTricks>13-greatest:

                print ("Congratulations, you played yourself. In other words, You took an L :) ")
                exit()
        else:

            if opponentTricks==greatest:

                print ("Congratulations, you played yourself. In other words, You took an L :) ")
                exit()

            elif yourTricks>13-greatest:

                print ("Congratulations, you beat this tarneeb")
                exit()


# play()
