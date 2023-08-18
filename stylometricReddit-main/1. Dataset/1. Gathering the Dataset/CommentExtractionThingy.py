#!pip install -q praw
import praw
import warnings

warnings.filterwarnings("ignore")

reddit = praw.Reddit(
    client_id="QHv39t6vcJE1zbV6dcyuhg",
    client_secret="AILcqPXTPlVY-o2Q0lEwSCMUuytf3g",
    user_agent="Comment Extraction (by u/SoC_CommentPull)",
    username="SoC_CommentPull",
    password="np.zTq^nLM8G!FT",  # all these are for API access
)

user = reddit.redditor("MagicCarpet5846")  # username

with open("MagicCarpet5846.txt", "w", encoding="utf-8") as file:
    for comment in user.comments.new(limit=None):
        file.write(comment.body + "\n")
