// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// GitHub Pages configuration
// For project sites: https://<org>.github.io/<repo>/
const site = process.env.SITE_URL || 'https://branes-ai.github.io';
const base = process.env.BASE_PATH || '/embodied-ai-architect';

// https://astro.build/config
export default defineConfig({
	site: site,
	base: base,
	integrations: [
		starlight({
			title: 'Embodied AI Architect',
			description: 'Design, analyze, optimize, and deploy embodied AI solutions',
			social: [
				{ icon: 'github', label: 'GitHub', href: 'https://github.com/branes-ai/embodied-ai-architect' },
			],
			logo: {
				light: './src/assets/logo-light.svg',
				dark: './src/assets/logo-dark.svg',
				replacesTitle: false,
			},
			editLink: {
				baseUrl: 'https://github.com/branes-ai/embodied-ai-architect/edit/main/docs-site/',
			},
			sidebar: [
				{
					label: 'Getting Started',
					items: [
						{ label: 'Introduction', slug: 'getting-started/introduction' },
						{ label: 'Installation', slug: 'getting-started/installation' },
						{ label: 'Quickstart', slug: 'getting-started/quickstart' },
					],
				},
				{
					label: 'Features',
					items: [
						{ label: 'Model Analysis', slug: 'features/model-analysis' },
						{ label: 'Hardware Selection', slug: 'features/hardware-selection' },
						{ label: 'Roofline Analysis', slug: 'features/roofline-analysis' },
						{ label: 'Constraint Checking', slug: 'features/constraint-checking' },
						{ label: 'Deployment', slug: 'features/deployment' },
					],
				},
				{
					label: 'Tutorials',
					autogenerate: { directory: 'tutorials' },
				},
				{
					label: 'Catalog',
					items: [
						{ label: 'Hardware', slug: 'catalog/hardware' },
						{ label: 'Models', slug: 'catalog/models' },
						{ label: 'Sensors', slug: 'catalog/sensors' },
						{ label: 'Operators', slug: 'catalog/operators' },
					],
				},
				{
					label: 'Reference',
					items: [
						{ label: 'CLI Commands', slug: 'reference/cli' },
						{ label: 'MCP Tools', slug: 'reference/mcp-tools' },
						{ label: 'Python API', slug: 'reference/api' },
						{ label: 'Constraints', slug: 'reference/constraints' },
					],
				},
				{
					label: 'Troubleshooting',
					autogenerate: { directory: 'troubleshooting' },
				},
			],
			customCss: ['./src/styles/custom.css'],
		}),
	],
});
