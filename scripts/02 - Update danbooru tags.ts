import Moepictures from "moepics-api"
import functions from "../functions/Functions"

const updateDanbooruTags = async () => {
    const moepics = new Moepictures(process.env.MOEPICTURES_API_KEY!)

    const tags = await moepics.tags.list([])

    let i = 0
    for (const tag of tags) {
        i++
        if (tag.type === "artist") continue
        console.log(i)
        let danbooruTag = tag.tag.replaceAll("-", "_")
        await moepics.tags.update(tag.tag, "danbooruTag", danbooruTag)
    }
}

export const updateDanbooruArtistTags = async () => {
    const moepics = new Moepictures(process.env.MOEPICTURES_API_KEY!)

    const tags = await moepics.tags.list([])

    let i = 0
    let skip = 0
    for (const tag of tags) {
        i++
        if (i < skip) continue
        if (tag.type !== "artist") continue
        console.log(i)
        let danbooruArtistTag = await functions.getDanbooruArtistTag(tag.tag)
        if (danbooruArtistTag) await moepics.tags.update(tag.tag, "danbooruTag", danbooruArtistTag)
    }
}

export default updateDanbooruTags