// Files copied from the repository (data/demo, phase2/outputs, results) by scripts/sync-data.mjs.
const base = `${import.meta.env.BASE_URL}repo/`;
export const repoUrl = (path: string) => base + path;
export async function repoText(path: string): Promise<string> {
  const res = await fetch(repoUrl(path));
  if (!res.ok) throw new Error(`Could not load ${path} from the repository copy. Run "npm run sync-data".`);
  return res.text();
}
export const repoJson = async <T,>(path: string): Promise<T> => JSON.parse(await repoText(path)) as T;
