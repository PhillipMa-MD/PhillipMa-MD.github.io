---
title: "Everything is Securities Fraud"
date: 2020-07-29T04:01:06+00:00
categories:
  - projects
tags:
  - Machine Learning
  - Finance
excerpt: "I train a GPT-2 model to produce articles in the style of Matt Levine's 'Money Stuff'"
---

## An overly-long story

A friend of mine turned me onto [Matt Levine](https://twitter.com/matt_levine) earlier this year for reasons I can no longer remember, but for which I am very grateful. Matt is a former lawyer/banker and currently a writer for Bloomberg where he puts out a lunchtime newsletter (almost) every weekday called [MoneyStuff](https://www.bloomberg.com/opinion/authors/ARbTQlRLRjE/matthew-s-levine) in which he provides his take on “Wall street, finance, companies and other stuff”. His writing style is uniquely approachable (given the subject matter) and he is in fact, hilarious. Reading the newsletter gives me (what I believe to be) at least a light finger on the pulse of finance which I otherwise find very difficult to keep up with. He also occasionally links to other writers/twitter people I follow which makes me feel validated.

In [January](https://www.bloomberg.com/opinion/articles/2020-01-07/no-rush-repaying-these-student-loans) and then again in [February](https://www.bloomberg.com/news/newsletters/2020-02-21/money-stuff-the-companies-are-on-the-same-team), Matt wrote about [GPT-2](https://openai.com/blog/better-language-models/) (OpenAI’s language generation tool/model) and how cool/interesting people were putting it to use in cool/interesting ways. See: [here](https://play.aidungeon.io/) and [here](https://web.archive.org/web/20200310063743/https://slatestarcodex.com/2020/01/06/a-very-unlikely-chess-game/). I had heard/read about machine learning in general and GPT specifically through headlines and the occasional doomsaying article, and I had played around with some of the text generators but it seemed like more of a diversion to me before Matt Levine brought it up (emphasis is mine): 

> [GPT-2] is widely used as a sort of internet toy to write fake texts based on a model; **you can put a paragraph of Money Stuff into GPT-2 and get back a new paragraph that kind of follows logically and kind of sounds like Money Stuff.**

and this:

> GPT-2 is an artificial-intelligence language model: You feed it a corpus of texts, and it gives you back some more texts that sound like they could have come from that corpus. **You train GPT-2 on a bunch of Money Stuff columns, it learns to write Money Stuff, I am out of a job.**

Challenge accepted? And so I sort of decided to take this on as a project to teach myself how to put together workable python code, and to dabble in ‘machine learning’. 

Despite my reputation among my older family members, I am not much of a computer person. I was never formally trained in computer science, nor do I use it for work really. I have a vague understanding of how computers work mostly through googling problems as they arise, youtube videos, and from a brief period in the late 90’s where I would putz around on my computer aimlessly after I had run out of time on my AOL trial disk. I took a computer programming class (Pascal) in highschool, I sort of taught myself how to make rudimentary programs on my TI-83, I made it halfway through an online C++ course in college, and recently I took a more involved online course in Python geared towards data science. I thought that given my background, putting together a Matt Levine GPT-2 bot would take me like, a week. Tops. Lol. 

I found a Matt Levine twitter [bot](https://twitter.com/MattLevineBot) and some of those tweets are very good, but I wanted to have GPT-2 make entire articles! Someone had already done this too, and they named it [Machine Levine](https://0xnf.github.io/posts/ml/machinelevine/)! But as far as I could find, the bot was no longer hosted and he didn’t provide any examples. I’m sure it worked just fine, but it didn’t use GPT-2, so I decided that my cause was righteous and set off. 

###### *Note: Why not GPT-3? When I began this project in February, it was not released. Also, at the point in time that I’m writing this (July 2020) GPT-3 was only available for poking and prodding to a small cohort of people who know what they’re doing. Also, it would likely require a whole cluster of TPUs to train which Google will almost certainly not let me play around with.*

I started off by attempting to gather every article of MoneyStuff I could find. I receive the newsletter in my inbox everyday (ish) so I figured I could start there, but I had only been reading him for a few months so this would have produced a relatively small amount of text for training data. Or so I read. I tried directly asking for the articles through BeautifulSoup, Urllib, and Selenium but as it turns out, Bloomberg’s web admins are smarter and better at their jobs of web-admin-ing than a random person on the internet who just ‘learned’ python; surprise suprise! I kept triggering their bot-detection which was frustrating but fair since it’d probably be bad if they just let random people bombard their servers all the time with abandon. I ended up running a pretty simple wget script which was super friendly to their servers, which means it took like 2 days to run. I also subscribed to bloomberg for one of those gimme introductory trials since I felt bad for circumventing their paywall. 

Next, I had to strip away the HTML and (and a bunch of other stuff I don’t understand) to get to the meat of the articles. My early attempts using various readability focused modules (newspaper, BS4’s built in gettext, html2text) were unsatisfactory since they tended to leave weird parts of the header in, or truncated parts of the article which I wanted to keep (or equally as likely, I just didn’t know how to use them properly). They also completely stripped the formatting which was disappointing because I sort of wanted GPT-2 to be able to generate full articles with titles, subtitles, and paragraphs (again, probably a way around this. Perhaps my stackoverflow-fu is lacking). I ended up frankensteining together enough script (using BeautifulSoup) from stackoverflow to prune out the navigation bar, ads, videos, pictures, and self-recommending links. Then I deployed Selenium to open each HTML file, select all the text, and paste the text into a text file. I think that my approach was a spectacularly dumb and inefficient way to do this and there is almost certainly an elegant way to do it with regular expressions (AAAHHHHH) or something in like 4 lines. But alas, I am bad at this. 

I do not own an enterprise-grade GPU, or any GPU at all for that matter. My Macbook Pro is about 8 years old, and is rocking a Core i5 with integrated graphics. This probably wasn’t going to cut it if I wanted to train even the smallest GPT-2 model in a reasonable amount of time. I googled around and found out that Max Woolf had created a [Colaboratory notebook](https://colab.research.google.com/drive/1VLG8e7YSEwypxU-noRNhsv5dW4NfTGce) with an attached GPU where you can run GPT-2 fine tuning for free! What a time to be alive. I settled on the 355M model (after reading up on what [Gwern](https://www.gwern.net/GPT-2) and [Shawn Presser](https://twitter.com/theshawwn) did earlier this year and in late 2019) because apparently this was the best I could accomplish using free Colaboratory notebooks. The colaboratory notebooks time out after 90 minutes and run for a maximum of 12 hours. So I wasted about half a week retraining the model after losing the checkpoint data on disconnect pretty much every day I tried it. So instead, I began just running it in batches everyday until I was satisfied. This is probably bad somehow, and there is likely a way to run fine tuning in Colab without babysitting, but you know. Whoops. 

I don’t think I can stress this enough, but I don’t know what I am doing. I did not learn anything about actual Machine Learning through this project. GPT-2 was basically a black box for me which I fed a text file. Actual ML is likely way more complicated than I can reasonably imagine, and I honestly probably don’t remember enough math to get into the nitty gritty. Regardless, it was fascinating to see it go brrrr and spit out funny articles, and it was super cool of smart people to release these (super simplified) tools to the masses to play with. Anyway, the way to keep track of how well your model is doing (apparently) is by reading the ‘labeled’ numbers the terminal shoots out at you. 

Google (mostly Gwern) tells me that ‘loss’ is Cross-Entropy Loss which sort of makes sense to me, but also it wouldn’t surprise me if it ended up meaning something else entirely. Avg is what I assume to be a running average of the loss which would show you a general trajectory of your model’s progress. As far as I could tell, this is only an average of the loss in the current training session. What I gleaned was that you’re supposed to read through the sample texts at the intervals you set (mine was every 500 steps) to check to see if the model is getting better, and also to check to see if your model’s loss is improving anymore (benefitting from further training) to know when to stop. In retrospect, I probably should have saved these terminal outputs and graphed the progress because it’d probably look cool and official to have that as a figure right about here. Oh well. 

## Anyway

Here are some (not cherry picked) examples:

*Note: The original post on squarespace had quite a few more examples but I was able to embed them in little scrolly boxes. If I ever figure out how to do that here, well, you wouldn't be reading this. Otherwise, I included just 2 so you can get the gist.*
>
>## Telling Banks to Slow Down and Listen to the Customers
>
>Also an insurance-company answer, market structure, the Mob Rule, and risk-based capital.
>
>### Telling banks to listen to customers.
>
>The basic thesis of complaints about modern market structure is that it is too focused on the short term. Long-term players want to invest in things that are easy to understand, that have obvious long-term implications. If some dumb trading 20-minute thing just won't work any more, you can tell them to go away, because that's where the value is. "Okay, I understand the concept of short-term risk-adjusted returns, but I am only human and can't reliably deliver those returns in the future." But if you are the global head of the New York Stock Exchange, you can promise your customers that you'll bring them those returns in the future, because you should. You should.
>
>There are two objections to this theory. One is that it is historically inaccurate. It doesn't really reflect the financial world any more than the rest of us, and it conflicts with the actual experience of business in modern markets. People invest in a lot of dumb stuff because it is intuitively obvious that it is stupid, and because dumb things are better for returns than the ones that are obvious to humans. But you have to make a distinction between the dumb stuff (bid/ask spreads, 20-minute volatility rides, uncertain pricing) and the interesting stuff (daily corporate access, daily photo collages). If your dumb bidding/ask spread is 40 times your long-term expected spread, you should go around telling companies to raise their capital because they're dumb, but you shouldn't be. If your long-term expected spread is eight times your dumb-but-inexpensive spread, you shouldn't even be investing in stocks.
>
>The other objection is that banks are just like other businesses. They go around advising companies on mergers, sure, but they also see lots of merger-related sales pitches. They sense that the company is interested in the long term, and it knows how to find the smartest people to invest in the right companies. But banks aren't in the business of, like, making the ultimate decision of how to go about mincing out the dumb bidders and into the companies. It's just taking a fair share of the market for itself. It's a private equity business. The thing that banks do for free is make good decisions, and then act like dumb jerks whenever it feels like they're being pocketed.
>
>The second objection is less dumb and more accurate. Banks are not in the business of investing in companies that go public before going public. This one is more accurate; lots of people think that. But it is not an iron law of nature. Nothing stops a bank from investing in a company that goes public, or even from doing a stock buyback. A company can go public without having quarterly earnings, can acquire public-market investors, can borrow money, etc. But a bank can't just create a company and go public; it has to have some "obligation to maintain, protect, and enforce its books, records, and other rights bylaws and rules," and those rights might be required just to keep its books, records etc. under the current regime. Banks are public companies with fiduciary duties to their shareholders. But as a lot of U.S. high-frequency traders have discovered, doing a stock buyback, or a merger buyback, or an exchange-traded fund buyback, or whatever, is a perfectly legal thing to do. A lot of them are pretty upset about it.
>
>### Anyway here's Dan McCrum at FT Alphaville with more on the Deutsche Bank buyback:
>
>The acquisition, which snapped a string of failed deals that has made Deutsche Bank one of Wall Street’s most profitable companies, has sent shares tumbling 13 per cent so far on the news that the capital analyst has stopped using the word ‘buy’ in the same sense in which “buy’ means buy.
>
>>“For a while the media was having a bit of a campaign to make us look like a buy-side bank,” said one person with direct knowledge of the discussions. “They tried to make us look like a buy-side bank by turning us into a buy-side bank.”
>
>The buyout, which was announced in March, was led by the unit of the New York-based bank, known as the Stand by Bearer. The unit, set up by Jeff Raich, will now be managed by the existing unit in a more hands-off manner.
>
>>"The original idea here was to partner with a US stock exchange to get the buy-side to buy and hold for longer," says the person. "The idea was

>## Blockchains, Ponzis and Data Loss
>
>Also Jamie Dimon, Merck, crime, stock buybacks, etc.
>
>### Blockchains.
>
>The more promising use case for distributed ledger technology is probably the one where it is widely used to keep and verify the Bitcoin blockchain, but with far less oversight and compliance issues. You just do your thing, and on the blockchain. So:
>
>>The owner of the world’s first cryptocurrency exchange has been shut down by police in China after being accused of money laundering, the Financial Times reported.
>>
>>The owner, who was “officially” investigated by the Federal Investigations Division of the China Justice Ministry and state media, was discovered by investigators at the Hangzhou police station just steps from the exporter’s office.
>>
>>A note on Yanping Lu, the official verified by the Financial Times, said Mr Lu was a “constitutive officer” and had appeared before the court.
>
>It is a bit unclear to me whether the person who was found in possession of the money was a high-ranking government official in charge of the Hangzhou Hangzhou Bitcoin Exchange, or just a random person who happens to own Hangzhou's first cryptocurrency. It seems unlikely that the person who was discovered to have the Hangzhou Hangzhou Bitcoin Exchange might still be in charge, though I suppose there is some precedent for prosecutors catching a bank and locking up its money, just out of laziness.
>
>Meanwhile here is a story about how "insiders poured millions into Bitcoin, believing it would end the need for a government-backed intermediary." And here is "a U.S. appellate judge has thrown out the government’s case against cryptocurrency purveyor Mt. Gox, declaring bitcoin a money-market fund," because all Bitcoin is money-market fund stuff now. I am currently searching for a card game to represent the Bitcoin crisis. I would strongly recommend "Cantabria Games Inc."
>
>### People are worried about unicorns.
>
>Here is my second-most-satisfying unicorn worry, which is the one about how the unicorns are doing this, like, five minutes after the Enchanted Forest is finished, and then at five minutes after ruins everywhere:
>
>>Once in the hills, the group would set up shop, with tables designed for one or two people each, and distribute small bundles of grasshoppers or something at a time. The group would then search for a place to stick 40 to 60 sticks with up to 30 to 40 sticks, covered with a thin layer of the glossy greenish substance that usually accompanies the bean sprouts, which are prized in China for their healing powers.
>"Imagine looking down the road and seeing a small private patch of wilderness with a couple of giant, colourful, oversized, potted alicorn trees," and that road is a thoroughfare for the way to Silicon Valley. "It’s a tiny speck in the global economy," says a venture capitalist. I bet they make a stop at Palo Alto.

>### People are worried about bond market liquidity.
>
>Here is a U.S. Treasury blog post from last week called "Regulating the Financial Intermediaries Act," intended to improve "the U.S. bond market by encouraging electronic trading and its use of underwriter and counterparty-based credit services." It's relevant to the current state of the bond market, as central-bank-driven deflation often leads to increased bids and offers, which leads to less trading, and to more settling of settled trades. But Treasury isn't concerned:
>
>>Although dealer banks would like to maximize bid-ask spreads, they are unable to do so because dealers have limited inventory and therefore rely on a large stock market capitalization to tie their balances.
>Instead, electronic trading firms "will provide liquidity by periodically quote quote quote and establish a bid-ask spread," to try to "tweak" dealers. I find myself a bit sympathetic to the idea that dealers are useful intermediaries, in fixed-income markets, but it always strikes me that they are just working in the shadows.
>
>Elsewhere: "Bond prices rally in the second week of May."
>
>### Things happen.
>
>Brexit Polls Pose Hurdles for Financial Advisers. Second Lien Finance Firms Face Off Against Short-Term Interest. AT&T Is Said to Pull Amid Comcast Merger Push. Ocwen’s Restructuring and Accounting Troubles Raise Questions Over Accounting Rules. Tesla Won’t Qualify as a Qualification Investor Under the Model 3. Twitter’s Jack Dorsey Wins NFL Player Of The Year award. "If 10% of people who have applied for jobs at the bank are like ‘I am not interested,’ then how do you sell that card company?

So, I’d say that at the very least, these are funny. I certainly thought so anyway. For context, [here](https://www.bloomberg.com/opinion/articles/2020-07-27/spacs-aren-t-cheaper-than-ipos-yet) is a random MoneyStuff article to compare his writing style and structure. I’d say the model does a pretty good job at getting the structure down. I added the formatting to these examples but the plain text generated clearly (through use of a new line) generates the article titles, subtitle, and section headings itself. I was most impressed with the model’s ability to emulate Matt’s unique style which really was the point of me making this so, yay. Unfortunately, GPT-2 seems incapable (or equally as likely, I am incapable) of generating texts long enough to match a typical MoneyStuff article which usually spans about 3-5 sections. Also, the section headings often don’t match the subtitles, although they do seem to at least track well with the main title. So if I were actually trying to use this to generate full Matt Levine articles I could probably piece a few together to make one that’s somewhat passable at a glance. I also apparently let slip a few instances of “email the editor here at” and early titles including “Matt Levine on Wallstreet” which you know, I guess adds authenticity to them but probably also prevented the model from getting to what’s really important; but I was impressed that it knew to always put the email stuff at the bottom so it learned something at least. 

Here are some things I learned:

- Be nice to websites and they will be nice to you.
- I don’t know what Machine Learning really is. I was given a kit-car and I ‘successfully’ put it together, but I still know nothing about how internal combustion engines function other than vague buzzwords (fuel injection, oxygen ratios, pistons, transmission). But now I have one and making it was fun, so. Yay. 
- People much smarter than I, who know more about what’s going on under the hood, also encountered many of the same difficulties that I did, so I needn’t feel that bad. 
- Everything takes longer than you think it should.
  
Next, I’ll probably try to continue training the bot I’ve got to a lower loss, just to see what comes out. This is apparently possible with the larger models (774M, 1.5B, GPT-3) and some people have figured out how to train these models (at least the first two) on the kind of hardware that google gives you for free. I’m also going to look into hosting the bot somehow so that you can prompt it or just make it generate an article on a day where Matt takes a day or so off. 

##### Footnote: Obviously, none of the above examples were written by Matt Levine, or any real person for that matter. They were ‘written’ by a ‘fine-tuned GPT-2 model’. Anything written about real people, places, companies, or things are almost certainly not true and should not be interpreted as such.

##### Footnote 2: I emailed this and some additional examples to Matt Levine and he included a section about it in today’s (July 30th) [MoneyStuff](https://www.bloomberg.com/opinion/articles/2020-07-30/kodak-is-relevant-again). This was very cool of him and I am pretty psyched that he thought it was entertaining and worthy of his column.