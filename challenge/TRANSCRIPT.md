# Kick-off transcript — TableSalt / Lerai challenge

Cleaned transcription of the kick-off call (`Transribe` in the repo root). Nothing was added;
filler words ("um", repetitions) were dropped, otherwise the wording is verbatim.

- **Speakers:** 1 = organizer/presenter (also answers the pitch questions), 2 = Samuel,
  3 = Mo, 4 = participant, 5 = Fahad (co-founder of Lerai, owns the problem), 6 = late arrival.
- **Unclear passages:** `[?]`. Likely transcription errors are bracketed: `tax [text] law` etc.
- **Open item from the call:** the repo link was posted to the WhatsApp channel; access is by
  "Use this template" → private repo → add `metinferhat@outlook.com` as collaborator.

---

## Introductions

**00:00:02 — S1**
Nice to meet you. My name is Elias.

**00:00:09 — S2**
I'm 18 years old. I'm studying a bachelor's majoring in data science in the first semester.
I started this year here. And I'm looking forward to the challenge.

**00:00:19 — S2** *(Samuel)*
Hello, I'm Samuel. I'm 24, not 25. Today is my birthday.

**00:00:24 — S3**
Oh happy birthday!

**00:00:25 — S2**
I'm studying applied artificial intelligence at [HHN] and I've worked on something similar before.

**00:00:33 — S3**
Really?

**00:00:37 — S3**
Hello, my name is Mo. I'm also studying management and data science first semester. I'm twenty
years old.

**00:00:44 — S4**
I feel like I'm twenty. I'm twenty-two, and I'm also studying management and data science first
semester.

**00:00:52 — S5**
I'm also Elias. I'm also studying Bachelor of Management and Data Science at TU. And I'm also 18,
so I'm a student. So how are the teams right now? We all three, and you guys three.

**00:01:05 — S3**
Or we were two and we have another third. He didn't show up so far. If he doesn't show up, we can
work with [without] him.

**00:01:13 — S3**
Okay.

## The problem (Fahad, Lerai)

**00:01:16 — S5**
My name is Fahad. I'm co-founder of Lerai [Derai]. I'm 29 years old. Currently working at both.
I'm trying to quit my job, but I'll just talk about our project because you were also asking
questions. And then we'll just jump into our presentation.

**00:01:30 — S5**
So what we do is basically — I do a brief explanation. We are creating an AI platform for tax
[text] consultations, and what that means is these guys are doing a lot of manual tasks in their
daily life, but this always needs some semantic attention, which means you cannot basically create
a deterministic algorithm because, for example, they have to create something called a
[annual] share performance [statement] at the end of each quarter.

**00:02:04 — S5**
They go over their client's documents. What they bought or purchased? What they invested in? Did
they fire someone? Did they buy new furniture for their office? And they create a report and send
this report to finance and to the government. Because you know that's better — [taxes are] probably
the most important thing in Germany, right?

**00:02:20 — S5**
And then there is this answer coming from the Finanzamt. They go over the documents again. They do
some stuff and make another report. And it's always like this, right? And if they don't like
something, they can take the case to the court, and court decides against them or in favor of them.

**00:02:43 — S5**
And then later they can use these court decisions to make their cases stronger. They give references
to these court decisions, and these court decisions actually update tax [text] law for clients.

**00:02:57 — S5**
And what we do is basically follow the tax law in general, court decisions, new regulations, BMF
[BMS] letters, whatever. But we also know the client data — so that's where you are going to mostly
work on.

**00:03:12 — S5**
Our customers have thousands of clients, and each client has tens of thousands of documents. And
they're sometimes just like a picture of a document uploaded, and it's just lost somewhere. Or a
[table] maybe 50 pages long.

**00:03:29 — S5**
And there's this problem with the tax consultants because Germany wasn't so much into
digitalization, or a bit slow. People were still — and actually there are still — doing a lot of
scanned documents. They are just like posting their invoices or whatever to tax analysis, and they
just take the photo, upload it to the system, they read it there.

**00:04:06 — S5**
But there's a problem: if you take the photo like this or in bad illumination, or the paper has
some wrinkles on it, then you cannot really extract the text in it.

**00:04:06 — S5**
Of course, you can just send the whole paper, whole picture, to the LLM and expect it to read.
Mostly, it does a good job, a perfect job, but it's way too expensive. You cannot just make it read
a fifty pages long PDF for a simple question.

**00:04:25 — S5**
So what we're going to do is extract the text out of this — let's say — not so good scanned
documents. That's the idea. Do you have any questions?

## Q&A: scope, content, output

**00:04:38 — S4**
Is the documents only text or are the tables as well? What is the content of this?

**00:04:43 — S5**
It is tables.

**00:04:44 — S4**
It's but both or just tables. Graphics as well or not?

**00:04:51 — S5**
Not graphics, only tables. Only tables with texts inside.

**00:04:54 — S4**
But also just normal text like paragraphs. Does it have legal signs or stuff like that? Because is
it just A B C and numbers or is it also?

**00:05:05 — S5**
It is close to real world scenarios. Sometimes the columns are structured differently, parent
columns, but the content is mostly Latin alphabet and German alphabet with umlauts and everything.
You are expected to normalize the content, which means — for example, the floating numbers. You are
using a comma instead of a point to represent a floating number. You should also make this
universal.

**00:05:38 — S5**
So there are some test cases. You will already see the tables themselves. But there will also be
hidden tables. I think there are twenty-ish open tables and ninety-ish hidden test cases.

**00:05:56 — S5**
So don't try to feed your algorithm on [overfit to] the open test cases. There are some other
surprises in the hidden test case.

**00:06:05 — S1**
Too bad. Let's give it a surprise.

**00:06:08 — S4**
What type of surprises?

**00:06:12 — S5**
Let's give it a surprise. Maybe just give it more information.

**00:06:16 — S5**
Maybe I can talk about one. For example, the tests you will be getting are mostly table-oriented
PDF files or PNG files. But in one example, there could be text around it. Just an example.

**00:06:33 — S5**
But of course, you don't have to [get] full points on all the hidden test cases. We are not
expecting you to do that. But I find this problem quite solvable. Not so open-ended.

## What is judged

**00:06:49 — S5**
And I believe you are going to do minimal vibe coding. So try to understand the idea, come up with
your own creativity, and document that. Because I think this will make the difference in our
competition. You're all going to fulfill many test cases. But at the end, the difference will be
how you approached the problem and how the performance is.

**00:07:11 — S5**
All of your solutions will be run on my laptop. We will also evaluate the performance, the
requirements and specifications of the laptop, which is already written in the readme file that I'm
going to give you.

**00:07:34 — S5**
You're just gonna push your code. I will download it. It has to work in Docker because I don't want
to spend hours just setting up your environment. If there are, of course, small things that I need
to fix before, it's totally fine. But don't expect me to install fifty software.

**00:07:55 — S5**
Because I have limited time to evaluate the results. So it has to be running in a Docker
environment.

**00:08:04 — S5**
Any questions?

## Prior art and existing approach

**00:08:06 — S3**
Did you try any approaches for solving this problem before?

**00:08:09 — S5**
We are using one of the most common approaches available. It's open source. It's called Tesseract.
Tesseract is developed by IBM in the seventies.

**00:08:27 — S5**
This is mostly computer vision approach, with I think in the latest versions combined with some
smaller elements — or AI models, I'm not sure.

**00:08:39 — S5**
But there are different approaches. You can connect elements to each other, connect, or train your
own elements. Connect deterministic functions with other stuff, or you can only try to train one
model. And also, there are some AI models that try to do this job only, like [vision] AI models. Do
your research about it.

**00:09:08 — S5**
Um.

## Data provenance

**00:09:10 — S3**
Which kind of documents again? From which department do you get them? Do you scan your documents?

**00:09:17 — S5**
The documents that we prepare are not available because, of course, we can't share their
information. But these are mostly generic tables.

**00:09:28 — S3**
I can say. Is it from only one source?

**00:09:33 — S5**
We create it.

## Expected output

**00:09:35 — S4**
What is the result supposed to be? Is it just the digitalized document, or is it supposed to be that
you can ask about the contents of an element in the table? What's supposed to be the outcome?

**00:09:47 — S5**
Perfect. You also find in the repo story [README] that there are test cases and the expected
outputs. You can create expected output in JSON or HTML format.

**00:09:58 — S4**
That's not what I mean. Do you want the contents to be digitized, like that? You have a PDF
document, and before it was a picture, and now it is like actual text. Or do you want just the
contents to be written, summarized in a text? Or what exactly?

**00:10:15 — S5**
We don't expect you to create another PDF. We expect you to read that scanned document and create a
JSON or HTML file with the content itself.

**00:10:24 — S2**
So to keep the representation of the document structure but as text based.

**00:10:31 — S5**
I'm ready. You decide how you want to represent it.

**00:10:39 — S2**
But the benchmark you're running is only to that step. So the format we can freely choose.

**00:10:50 — S5**
But I recommend you changing more [choosing something sane]. But.

## Hand-in mechanics

**00:10:59 — S5**
If you already want it, I can give you the repository. It's public already. All you have to do is
use it as a template, create your own private repository, and add me as a contributor so I can read
your comments.

**00:11:04 — S5**
I'm just posting it on the WhatsApp channel.

**00:11:27 — S4**
Do we have the Codex credits already, or how do we get them?

**00:11:32 — S1**
Tomorrow.

**00:11:33 — S2**
I think.

**00:11:38 — S3**
So you're gonna give us pre-scanned documents, right?

**00:11:49 — S2**
I just don't know.

## Team logistics and closing advice

**00:11:59 — S1**
And if the other guys don't join, this is contact [?].

**00:12:03 — S3**
I think we have to take a decision in an hour to seven at the latest. He should be here.

**00:12:12 — S2**
If Ahmed doesn't show up —

**00:12:16 — S6**
You've got a good laptop.

**00:12:30 — S6**
Haha. And why is it so hard?

**00:12:43 — S5**
It is not hard.

**00:12:44 — S1**
As I said, I expect you all to fix everything for the others. So I just — I just want to see how we
approach problems.

**00:12:53 — S2**
So we are probably all going to compete or beat the benchmark. But based on, for example,
performance or performance.

**00:13:02 — S1**
I don't share the idea.

**00:13:05 — S1**
What I would do — if I was in your place, I would just go and check out the performance first. And
try to understand, ask AI to try to understand why [OCR] is so hard, why it's failing, why tables
are that way or not that way. And then try to — solve each problem one by one instead of just AI
actually going and fixing this.

**00:13:39 — S2**
This is also what you meant, with in the presentation it said our reasoning behind different
decisions. At the end, you want something like an architectural decision record.

**00:13:55 — S1**
Because I can also check it up there. [Unrelated aside about creating a workflow for a customer
without content for it:] "to try to measure things, I try to give some concrete example, concrete
challenge. I hope you like it."

**00:14:23 — S1**
Good.

## After the break

**00:17:34 — S1**
And maybe it would be great if you can show me some intermediate results.

**00:17:42 — S1**
Help me maximize better. Sure. Please.

**00:17:53 — S1**
Not until that time.

**00:17:55 — S5**
Today. Let me check.

**00:17:58 — S1**
Program.

**00:18:01 — S1**
You guys here on the game?

**00:18:16 — S1**
Probably around eight.

**00:18:18 — S1**
But like, I was to be like —

**00:18:20 — S5**
I just need to drive to Costa Rica [unclear].

**00:19:23 — S6**
Everything.

**00:20:25 — S6**
620° [unclear].

---

## Terms that were probably mistranscribed

| In transcript | Almost certainly |
|---|---|
| Derai | Lerai (the company; repo org `lerrai-gmbh`) |
| text consultations / text law / text analysis | tax consultations / tax law / tax analysis |
| BMS letters | BMF letters (*BMF-Schreiben*) |
| share performance | Jahres-/Quartalsabschluss, "share [annual] report" |
| exam table / 50 pages long | a table, maybe 50 pages long |
| white coding → "minimal white coding" | vibe coding |
| "why Ollama is so hard" | why OCR (or "why it") is so hard |
| Hahn | HHN — Hochschule Heilbronn |
| Shaberta | unclear, customer example |
