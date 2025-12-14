import Moepictures from "moepics-api"
import functions from "../functions/Functions"
import child_process from "child_process"
import util from "util"
import path from "path"
import fs from "fs"

const exec = util.promisify(child_process.exec)

const generateOCRNotes = async () => {
    const moepics = new Moepictures(process.env.MOEPICTURES_API_KEY!)

    const posts = await moepics.search.posts({query: "+untranslated +partially-translated", type: "image", rating: "all+h", style: "all+s", sort: "reverse date", limit: 99999})
    //const posts = await moepics.search.posts({query: "-translated", type: "comic", rating: "all+h", style: "all+s", sort: "reverse date", limit: 99999})
    console.log(posts.length)
  
    let i = 0
    let skip = 0
    for (const post of posts) {
        i++
        if (Number(post.postID) < skip) continue
        let updated = false
        for (const image of post.images) {
            console.log(`${i} -> ${post.postID} / ${image.order}`)
            let imageLink = moepics.links.getImageLink(image, false)
            const buffer = await moepics.api.fetch(imageLink).then((r) => r.arrayBuffer())

            let imagePath = await functions.dumpImage(Buffer.from(buffer))
            const scriptPath = path.join(__dirname, "../../ocr/ocr.py")
            let command = `python3 "${scriptPath}" -i "${imagePath}"`
            const str = await exec(command).then((s: any) => s.stdout).catch((e: any) => e.stderr)
            fs.unlinkSync(imagePath)

            const data = JSON.parse(str.match(/(?<=>>>JSON<<<)([\s\S]*?)(?=>>>ENDJSON<<<)/gm)?.[0])

            if (data.length) {
                if (data[0].translation.includes("MYMEMORY WARNING")) return console.log(data)
                await moepics.notes.edit({postID: post.postID, order: image.order, data})
                updated = true
            }
        }
        if (updated) {
            await moepics.posts.removeTags(post.postID, ["untranslated", "partially-translated"])
            await moepics.posts.addTags(post.postID, ["translated", "notecheck"])
        }
    }
}

export default generateOCRNotes