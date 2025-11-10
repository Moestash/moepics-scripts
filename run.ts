import "dotenv/config"
import updateTwitterMirrors from "./scripts/9 - Update twitter mirrors"

const start = async () => {
    updateTwitterMirrors()
}

start()