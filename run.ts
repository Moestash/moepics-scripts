import "dotenv/config"
import mergeDuplicateArtists from "./scripts/5 - Merge duplicate artists"

const start = async () => {
    mergeDuplicateArtists()
}

start()