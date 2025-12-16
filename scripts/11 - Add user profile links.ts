import Moepictures from "moepics-api"
import Pixiv, {PixivIllust} from "pixiv.ts"

const addUserProfileLinks = async () => {
    const moepics = new Moepictures(process.env.MOEPICTURES_API_KEY!)
    const pixiv = await Pixiv.refreshLogin(process.env.PIXIV_REFRESH_TOKEN!)

    const posts = await moepics.search.posts({query: "", type: "image", rating: "all+l", style: "all+s", sort: "reverse date", limit: 99999})
  
    let i = 0
    let skip = 0
    for (const post of posts) {
        i++
        if (i < skip) continue
        if (post.userProfile) continue
        console.log(i)
        if (post.source?.includes("pixiv.net")) {
            const id = post.source.match(/\d+/)?.[0] ?? ""
            let illust: PixivIllust
            try {
                illust = await pixiv.illust.get(id)
            } catch (e: any) {
                if (String(e)?.includes("access_token")) return console.log("429")
                continue
            }
            let userLink = `https://www.pixiv.net/users/${illust.user.id}`
            await moepics.posts.update(post.postID, "userProfile", userLink)
            await moepics.posts.update(post.postID, "drawingTools", illust.tools)
            await moepics.posts.update(post.postID, "bookmarks", illust.total_bookmarks)
            await moepics.posts.update(post.postID, "sourceImageCount", illust.page_count)
        }
    }
}

export default addUserProfileLinks